"""
This module implements the Lowess function for nonparametric regression.
Functions:
lowess Fit a smooth nonparametric regression curve to a scatterplot.
For more information, see
William S. Cleveland: "Robust locally weighted regression and smoothing
scatterplots", Journal of the American Statistical Association, December 1979,
volume 74, number 368, pp. 829-836.
William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
approach to regression analysis by local fitting", Journal of the American
Statistical Association, September 1988, volume 83, number 403, pp. 596-610.
"""

# Originally based on https://gist.github.com/agramfort/850437
# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
# License: BSD (3-clause)

import os
from datetime import datetime

import numpy as np
import pymp
from matplotlib.dates import date2num
from numba import njit, prange

from apertools.log import get_log, log_runtime
from apertools.utils import block_slices

logger = get_log()


def lowess_stack(stack, x, frac=0.4, min_x_weighted=None, n_iter=2, n_jobs=-1):
    """Smooth a stack of images using linear lowess.

    When n_iter > 1, will rerun each regression, reweighting by residuals
    and clipping to 6 MAD standard deviations.

    Args:
        stack (ndarray): stack with shape (ns, nrows, ncols)
        x (ndarray): x values with shape (ns,)
        frac ([type], optional): fraction of data to use for each smoothing
            Defaults to 2/3. Alternative to `min_x_weighted`.
        min_x_weighted (float, optional): Minimum time period of data to include in smoothing
            in the units of x.
            Alternative to `frac`. Defaults to None.
        n_iter (int, optional): Number of iterations to rerun fit after
            reweighting by residuals.
            Defaults to 2.
        n_jobs (int, optional): Number of parallel processes to use.
            Defaults to -1 (all CPU cores).

    Returns:
        ndarray: stack smoothed with shape (ns, nrows, ncols), smoothed along
            the first dimension.
    """
    if not stack.dtype.isnative:
        stack = stack.astype(stack.dtype.type)
    if n_jobs < 0:
        n_jobs = os.cpu_count()
    if stack.ndim == 1:
        stack = stack[:, np.newaxis, np.newaxis]
    if stack.ndim != 3:
        raise ValueError("stack must be 1D or 3D")

    ns, nrows, ncols = stack.shape

    # in_stack = np.moveaxis(stack, 0, 2)
    if min_x_weighted:
        frac = find_frac(x, min_x_weighted)

    # Make each depth-wise pixel into a column (so there's only 1 OMP loop)
    stack_cols = stack.reshape((ns, -1))
    # stack_out = np.zeros(stack_cols.shape, dtype=stack.dtype)
    # for index in range(0, stack_cols.shape[1]):
    n_pixels = stack_cols.shape[1]
    stack_out = pymp.shared.array((ns, n_pixels), dtype=stack.dtype)
    with pymp.Parallel(n_jobs) as p:
        for index in p.range(0, n_pixels):
            y = stack_cols[:, index]
            stack_out[:, index] = lowess_pixel(y, x, frac, n_iter)

    return stack_out.reshape((ns, nrows, ncols))

    # reshaping so it's contiguous in last dimension
    # out = np.zeros((rows, cols, ns), dtype=stack.dtype)
    # _numba_loop(in_stack, x, frac, n_iter, out)
    # return np.moveaxis(out, 2, 0)


@njit(nogil=True, parallel=True)
def _numba_loop(in_stack, x, frac, n_iter, out):
    for r in prange(in_stack.shape[0]):
        for c in range(in_stack.shape[1]):
            y = in_stack[r, c, :]
            out[r, c, :] = lowess_pixel(y, x, frac, n_iter)


@njit(nogil=True)
def lowess_pixel(y, x, frac=0.4, n_iter=2, do_sort=False, x_out=None):
    """Run LOWESS smoothing on a single pixel.

    Note that lowess smoother requiring native endian datatype (for numba).
    Performs multiple iterations for robust fitting, downweighting
    by the residuals of the previous fit.

    Args:
        y (ndarray): y values with shape (ns,)
        x (ndarray): x values with shape (ns,)
        frac ([type], optional): fraction of data to use for each smoothing
            Defaults to 2/3.
        n_iter (int, optional): Number of iterations to rerun fit after
            reweighting by residuals.
            Defaults to 2.
        do_sort (bool, optional): Sort x and y before running LOWESS.
            Defaults to False.
        x_out (ndarray, optional): x values to output. If None, will use x.

    Returns:
        ndarray: smoothed y values with shape (ns,)
    """
    n = len(x)
    if x_out is None:
        x_out = x
    # x_out = np.ascontiguousarray(x_out)

    if np.any(np.isnan(y)) or np.all(y == 0.0):
        return np.zeros(len(x_out))

    if do_sort:
        # Sort the x and y values
        sort_idxs = np.argsort(x)
        x = x[sort_idxs]
        y = y[sort_idxs]

    # First run LOWESS on the data points to get the weights of the data points
    # using it-1 iterations, last iter done next
    if n_iter > 1:
        _, delta = _lowess(y, x, np.ones_like(x), frac=frac, n_iter=n_iter - 1, x_out=x)
    else:
        # Initialize the residual-weights to 1
        delta = np.ones(n)
    # print(np.sum(delta))
    # Then run once more using those supplied weights at the points provided by xvals
    # No extra iterations are performed here since weights are fixed
    y_out, _ = _lowess(y, x, delta, frac=frac, n_iter=1, x_out=x_out)

    return y_out


@njit(cache=True, nogil=True)
def _lowess(y, x, delta, frac=0.4, n_iter=2, x_out=None):
    """Actual linear fit loop, given the starting residual weights"""
    n = len(x)
    n_out = len(x_out)
    r = int(np.ceil(frac * n))
    if r == n:
        r -= 1

    # Get the matrix of local weights for each step
    # xc = np.ascontiguousarray(x)  # make contiguous (necessary for `reshape` for numba)
    xc = x

    # Find the distance to the rth furthest point from each x value
    h = np.array([np.sort(np.abs(xc - xc[i]))[r] for i in range(n)])
    # Get the relative distance to the rth furthest point
    # w = np.abs((xc.reshape((-1, 1)) - xc.reshape((1, -1))) / h)
    w = np.abs((xc.reshape((-1, 1)) - x_out.reshape((1, -1))) / h.reshape((-1, 1)))
    # Clip to 0, 1 (`np.clip` not available in numba)
    w = np.minimum(1.0, np.maximum(w, 0.0))
    # tricube weighting
    w = (1 - w**3) ** 3

    yest = np.zeros(n_out)
    for _ in range(n_iter):
        for i in range(n_out):
            weights = delta * w[:, i]
            # Form the linear system as the reduced-size version of the Ax=b:
            # A^T A x = A^T b
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([
                [np.sum(weights), np.sum(weights * x)],
                [np.sum(weights * x), np.sum(weights * x * x)],
            ])

            beta = np.linalg.lstsq(A, b)[0]
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        if s < 1e-3:
            break
        # delta = np.clip(residuals / (6.0 * s), -1.0, 1.0)
        delta = np.minimum(1.0, np.maximum(residuals / (6.0 * s), -1.0))
        delta = (1 - delta**2) ** 2

    return yest, delta


def lowess_xr(da, x_dset="date", min_days_weighted=2 * 365.25, frac=0.7, n_iter=2):
    """Run lowess on a DataArray stack.

    Args:
        da (xr.DataArray): 3D xarray containing data to be smoothed along dimension `x_dset`.
        x_dset (str, optional): Name of the time dimension. Defaults to "date".
        min_days_weighted (float, optional): Minimum time period of data to include in smoothing.
            See notes. Defaults to 365.25*2 (2 years of data).
        n_iter (int, optional): Number of LOWESS iterations to run to exclude outliers.
            Defaults to 2.

    Returns:
        xr.DataArray: stack from `da` smoothed along the dimension `x_dset`.

    Notes:
        When sampling is irregular, specifying one fraction of data for lowess will lead to parts
        of the smoothing using longer time intervals. `min_days_weighted` is used to specify the
        minimum time desired for the smoothing. For example, if the data starts as sampled every
        month, but then switches to sampling every 2-weeks, the fraction will be use the proportion
        of data that is needed to include at least `min_days_weighted` days of data during the
        2-week sampling time.
    """
    import xarray as xr

    x = date2num(da[x_dset].values)
    if min_days_weighted and min_days_weighted > 0:
        frac = find_frac(x, min_days_weighted)
    out_stack = lowess_stack(da.values, x, frac, n_iter)
    # Now return as a DataArray
    out_da = xr.DataArray(out_stack, coords=da.coords, dims=da.dims)
    out_da.attrs["description"] = "Lowess smoothed stack"

    out_da = _write_attrs(out_da, frac=frac, n_iter=n_iter)
    return out_da


@log_runtime
def lowess_mintpy(
    filename,
    in_dset="timeseries",
    out_dset="lowess",
    min_days_weighted=2 * 365.25,
    frac=0.7,
    n_iter=2,
    overwrite=False,
    block_shape=(256, 256),
    n_jobs: int = -1,
):
    """Run lowess on a MintPy timeseries file.

    Writes output to `out_dset` in `filename`.

    Args:
        filename (str): Path to MintPy timeseries file.
        min_days_weighted (float, optional): Minimum time period of data to include in smoothing.
            See notes. Defaults to 365.25*2 (2 years of data).
        n_iter (int, optional): Number of LOWESS iterations to run to exclude outliers.
            Defaults to 2.
        overwrite (bool, optional): Whether to overwrite existing /lowess dataset. Defaults to False.
        block_shape (tuple, optional): Shape of blocks to read from file. Defaults to (256, 256).
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to -1 (all cores).

    """
    # TODO:
    # record attrs used to lowess smooth
    # copy attrs from original dataset
    # add x/y geocoding (if available)
    import h5py

    with h5py.File(filename, "r+") as fid:
        if out_dset in fid:
            if overwrite:
                del fid[out_dset]
            else:
                raise ValueError(
                    f"{out_dset} already exists in {filename}. Use `overwrite=True` to"
                    " overwrite."
                )
        # Get the date strings
        dates = np.array(fid["date"][:]).astype(str)
        # convert to datetimes
        dates = [datetime.strptime(d, "%Y%m%d") for d in dates]
        # Convert to days
        x = np.array(date2num(dates))
        # Find the fraction of data to use
        if min_days_weighted and min_days_weighted > 0:
            frac = find_frac(x, min_days_weighted)

        shape = fid[in_dset].shape
        fid.create_dataset(out_dset, shape=shape, dtype=fid[in_dset].dtype, chunks=True)

        logger.info(
            f"Running lowess on {filename} with {n_jobs} jobs, {n_iter} iterations, and"
            f" {frac} fraction of data"
        )

        # Iterate in blocks using block_slices
        for rows, cols in block_slices(shape[-2:], block_shape=block_shape):
            logger.info(f"Running lowess on block {rows} {cols}")
            cur_block = fid[in_dset][:, slice(*rows), slice(*cols)]

            # Run lowess
            out_stack = lowess_stack(
                cur_block, x, frac=frac, n_iter=n_iter, n_jobs=n_jobs
            )
            # Write to file
            fid[out_dset][:, slice(*rows), slice(*cols)] = out_stack


def find_frac(x, min_x_weighted=None, how="all", min_points=50):
    """Find fraction of data to use so all windows are at least `min_x_weighted` days

    Args:
        x (ndarray): array of days from date2num
        min_x_weighted (int, float): Minimum time period (in units of `x`) to include in lowess fit
        max_x_weighted (int, float): Maximum time period (in units of `x`) to include in lowess fit
        how (str, optional): Options are "any" (default), "all".
            "any" means that any x difference can be above `min_x_weighted`.
            "all" means that all x differences must be above `min_x_weighted`. "all" produces
            larger fractions than "any"
        min_points (int, optional): Minimum number of points to include in lowess fit.
            Overrides `min_x_weighted` if `min_points` is greater than than `min_x_weighted`.

    Notes on min_x_weighted/max_x_weighted:
        For irregular sampling, the fraction of data to use can be based on the lower or
        high sample frequency parts of the data.

        For example, if x = [0, 3, 6, 9, 12, 13, 14, 15] days,
        and you want to use a window covering at least 4 days, then a 4-day window
        at the beginning (when the spacing is 3-days apart) covers 2 points,
        while at the end (when the spacing is 1-day apart) covers 4 points.
            min_x_weighted = 4 leads to a fraction of 2/8 = 0.125
            max_x_weighted = 4 leads to a fraction of 4/8 = 0.5

    """
    n = len(x)
    if not min_points:
        min_points = 0.0
    min_frac = np.clip(min_points / n, 0.0, 1.0)

    day_diffs = np.abs(x.reshape((-1, 1)) - x.reshape((1, -1)))
    # The `kk`th diagonal will contain the time differences when using points `kk` indices apart
    # (e.g. the 2nd diagonal contains how many days apart are x[2]-x[0], x[3]-x[1],...)

    if how == "any":
        # 'any' means that there's at least one point that is at least `min_x_weighted` days apart
        diffs = np.array([np.max(np.diag(day_diffs, k=kk)) for kk in range(n)])
    elif how == "all" or how is None:
        diffs = np.array([np.min(np.diag(day_diffs, k=kk)) for kk in range(n)])
    else:
        raise ValueError("how must be 'any' or 'all'")

    # Get the first diagonal where it crosses the min_x_weighted threshold
    idxs_larger = np.where(diffs > min_x_weighted)[0]
    if len(idxs_larger) > 0:
        return max(idxs_larger[0] / n, min_frac)
    else:
        return 1.0

    # Failed to make this work in parallel...
    # TODO: what does this even mean??
    # ValueError: Dimension `'__loopdim1__'` with different lengths in arrays
    # return xr.apply_ufunc(
    #     _run_pixel,
    #     da.chunk({"lat": 10, "lon": 10}),
    #     times,
    #     frac,
    #     n_iter,
    #     input_core_dims=[["date"], [], [], []],
    #     output_core_dims=[["date"]],
    #     dask="parallelized",
    #     output_dtypes=["float64"],
    #     exclude_dims=set(("date",)),
    #     dask_gufunc_kwargs=dict(allow_rechunk=True),
    #     vectorize=True,
    # )


@njit(cache=True, nogil=True)
def bootstrap_mean_std(y, x, frac=0.3, n_iter=2, K=100, pct_bootstrap=1.0):
    """Bootstrap mean and standard deviation of y

    Args:
        y (ndarray): y values
        x (ndarray): x values
        frac (float, optional): fraction of data to use for lowess fit. Defaults to 0.3.
        n_iter (int, optional): number of LOWESS iterations to run to exclude outliers.
            Defaults to 2.
        K (int, optional): number of bootstrap samples to take. Defaults to 100.
        pct_bootstrap (float, optional): fraction of data to use for bootstrapping.
            Defaults to 1.0.

    Returns:
        tuple: (mean, std) calculated from the bootstrapped samples.
        Shape of both `mean` and `std` are `len(y)`

    """
    bootstraps = bootstrap_lowess(
        y,
        x,
        frac=frac,
        n_iter=n_iter,
        K=K,
        pct_bootstrap=pct_bootstrap,
    )
    mean = np_nanmean(bootstraps, axis=0)
    std = np_nanstd(bootstraps, axis=0)
    return mean, std


@njit(cache=True, nogil=True)
def bootstrap_lowess(y, x, frac=0.3, n_iter=2, K=100, pct_bootstrap=1.0):
    """Repeatedly run lowess on a pixel, bootstrapping the data

    Args:
        y (ndarray): y values
        x (ndarray): x values
        frac (float, optional): fraction of data to use for lowess fit. Defaults to 0.3.
        n_iter (int, optional): number of iterations to run lowess. Defaults to 2.
        K (int, optional): number of bootstrap samples to take. Defaults to 100.
        pct_bootstrap (float, optional): fraction of data to use for bootstrapping. Defaults to 1.0.

    Returns:
        ndarray: bootstrapped lowess fits. Shape is (K, len(x))
    """
    out = np.zeros((K, len(x)))

    nboot = int(pct_bootstrap * len(x))

    for k in range(K):
        # Get a bootstrap sample
        # Note that Numba does not allow RandomState to be passed
        sample_idxs = np.random.choice(len(x), nboot, replace=True)
        # Need sort, since they will be mixed up from the bootstrap sample
        sample_idxs.sort()
        y_boot = y[sample_idxs]
        x_boot = x[sample_idxs]

        # Run lowess on the bootstrap sample
        out[k, :] = lowess_pixel(y_boot, x_boot, frac=frac, n_iter=n_iter)

    return out


def plot_bootstrap(
    x=None,
    y=None,
    frac=None,
    K=None,
    pct_bootstrap=None,
    xplot=None,
    mean=None,
    std=None,
    ax=None,
):
    import matplotlib.pyplot as plt

    if mean is None or std is None:
        if K is None:
            K = 100
        if pct_bootstrap is None:
            pct_bootstrap = 1.0
        if frac is None:
            frac = 0.4
        mean, std = bootstrap_mean_std(
            x, y, K=K, pct_bootstrap=pct_bootstrap, frac=frac
        )

    xplot = xplot if xplot is not None else x
    if xplot is None:
        xplot = np.arange(len(y))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.fill_between(xplot, mean - 1.96 * std, mean + 1.96 * std, alpha=0.25)
    ax.plot(xplot, mean, color="red")
    if y is not None:
        ax.plot(xplot, y, ".-")
    title = "Bootstrap mean and std "
    if frac is not None:
        title += f"frac={frac:.2f}"
    if K is not None:
        title += f" K={K}"
    if pct_bootstrap is not None:
        title += f" pct_bootstrap={pct_bootstrap:.2f}"
    ax.set_title(title)
    return fig, ax


@njit
def np_apply_along_axis(func1d, axis, arr):
    """Hack from https://github.com/numba/numba/issues/1269 to get around `axis` problem"""
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@njit
def np_mean(array, axis):
    """Same as np.mean, but with `axis` argument"""
    return np_apply_along_axis(np.mean, axis, array)


@njit
def np_std(array, axis):
    """Same as np.std, but with `axis` argument"""
    return np_apply_along_axis(np.std, axis, array)


@njit
def np_nanmean(array, axis):
    """Same as np.nanmean, but with `axis` argument"""
    return np_apply_along_axis(np.nanmean, axis, array)


@njit
def np_nanstd(array, axis):
    """Same as np.nanstd, but with `axis` argument"""
    return np_apply_along_axis(np.nanstd, axis, array)


# TODO: deduplicate this with the one in up top
@log_runtime
def bootstrap_lowess_xr(
    da,
    x_dset="date",
    min_days_weighted=2 * 365.25,
    frac=0.4,
    n_iter=2,
    n_jobs=-1,
    K=100,
    pct_bootstrap=1.0,
    out_fname=None,
    out_dsets=["defo_lowess", "defo_lowess_std"],
):
    """Get bootstrap estimates lowess mean and standard deviation of every pixel in a stack

    Args:
        da (xr.DataArray): 3D xarray containing data to be smoothed along dimension `x_dset`.
        x_dset (str, optional): Name of the time dimension. Defaults to "date".
        min_days_weighted (float, optional): Minimum time period of data to include in smoothing.
            See notes. Defaults to 365.25*2 (2 years of data).
        frac ([type], optional): fraction of data to use for each smoothing
            Defaults to 2/3. Alternative to `min_days_weighted`.
        n_iter (int, optional): Number of iterations to rerun fit after
            reweighting by residuals.
            Defaults to 2.
        n_jobs (int, optional): Number of parallel processes to use.
            Defaults to -1 (all CPU cores).
        K (int, optional): Number of bootstrap samples to draw.
            Defaults to 100.
        pct_bootstrap (float, optional): Percent of data to use for bootstrap.
            Defaults to 1.0.

    Returns:
        ndarray: stack smoothed with shape (ns, nrows, ncols), smoothed along
            the first dimension.
    """
    import xarray as xr

    x = date2num(da[x_dset].values)
    stack = da.values
    if min_days_weighted and min_days_weighted > 0:
        frac = find_frac(x, min_days_weighted)

    if n_jobs < 0:
        n_jobs = os.cpu_count()
    if stack.ndim == 1:
        stack = stack[:, np.newaxis, np.newaxis]
    if stack.ndim != 3:
        raise ValueError("stack must be 1D or 3D")

    ns, nrows, ncols = stack.shape
    # Make each depth-wise pixel into a column (so there's only 1 OMP loop)
    stack_cols = stack.reshape((ns, -1))

    if min_days_weighted:
        frac = find_frac(x, min_days_weighted)

    # stack_out = np.zeros(stack_cols.shape, dtype=stack.dtype)
    # for index in range(0, stack_cols.shape[1]):
    mean_cols = pymp.shared.array(stack_cols.shape, dtype=stack.dtype)
    std_cols = pymp.shared.array(stack_cols.shape, dtype=stack.dtype)
    with pymp.Parallel(n_jobs) as p:
        # for index in p.range(0, 3000):
        for index in p.range(0, stack_cols.shape[1]):
            y = stack_cols[:, index]
            mean, std = bootstrap_mean_std(
                x, y, frac=frac, n_iter=n_iter, K=K, pct_bootstrap=pct_bootstrap
            )
            mean_cols[:, index] = mean
            std_cols[:, index] = std

    stack_out_mean = mean_cols.reshape((ns, nrows, ncols))
    stack_out_std = std_cols.reshape((ns, nrows, ncols))
    # Now return as a DataArray
    out_mean = xr.DataArray(stack_out_mean, coords=da.coords, dims=da.dims)
    out_std = xr.DataArray(stack_out_std, coords=da.coords, dims=da.dims)
    out_mean.attrs["description"] = "Bootstrap mean of lowess smoothed stack"
    out_std.attrs["description"] = (
        "Bootstrap standard deviation of lowess smoothed stack"
    )

    out_mean = _write_attrs(
        out_mean, K=K, frac=frac, n_iter=n_iter, pct_bootstrap=pct_bootstrap
    )
    out_std = _write_attrs(
        out_std, K=K, frac=frac, n_iter=n_iter, pct_bootstrap=pct_bootstrap
    )

    # TODO: If there's an existing mean/std, update using
    # https://math.stackexchange.com/questions/374881/recursive-formula-for-variance or
    # https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
    if out_fname:
        mean_name, std_name = out_dsets
        out_ds = out_mean.to_dataset(name=mean_name)
        out_ds[std_name] = out_std
        logger.info("Saving %s/%s and %s/%s", out_fname, mean_name, out_fname, std_name)
        out_ds.to_netcdf(out_fname, mode="a", engine="h5netcdf")
    return out_mean, out_std


def _write_attrs(da, K=None, n_iter=None, frac=None, pct_bootstrap=None):
    """Write attributes to a DataArray"""
    attr_names = []
    attr_values = []
    if K:
        attr_names.append("bootstrap_replications")
        attr_values.append(K)
    if n_iter:
        attr_names.append("lowess_iterations")
        attr_values.append(n_iter)
    if frac:
        attr_names.append("lowess_fraction")
        attr_values.append(frac)
    if pct_bootstrap:
        attr_names.append("pct_bootstrap")
        attr_values.append(pct_bootstrap)

    for attr_name, val in zip(attr_names, attr_values):
        da.attrs[attr_name] = val
    return da


### Demonstrations of LOWESS steps ###


def demo_window(x, frac=0.4, min_x_weighted=None):
    if min_x_weighted:
        frac = find_frac(x, min_x_weighted, min_points=None)
    n = len(x)
    r = int(np.ceil(frac * n))
    if r == n:
        r -= 1

    # Find the distance to the rth furthest point from each x value
    h = np.array([np.sort(np.abs(x - x[i]))[r] for i in range(n)])
    # abs_dists = [np.abs(x - x[i]) for i in range(n)]
    # # h = np.array([np.sort(abs_dists[i])[r] for i in range(n)])

    xc = x.copy()  # make contiguous (necessary for `reshape` for numba)
    # Get the relative distance to the rth furthest point
    w = np.abs((xc.reshape((-1, 1)) - xc.reshape((1, -1))) / h)
    # Clip to 0, 1 (`np.clip` not available in numba)
    w = np.minimum(1.0, np.maximum(w, 0.0))
    # tricube weighting
    w = (1 - w**3) ** 3
    return w


def demo_fit(x, y, w, i, delta=None):
    if delta is None:
        delta = np.ones(len(x))
    weights = delta * w[:, i]
    # Form the linear system as the reduced-size version of the Ax=b:
    # A^T A x = A^T b
    b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
    A = np.array([
        [np.sum(weights), np.sum(weights * x)],
        [np.sum(weights * x), np.sum(weights * x * x)],
    ])

    beta = np.linalg.lstsq(A, b)[0]
    return beta


def demo_residual(x, y, w, i, delta=None):
    if delta is None:
        delta = np.ones(len(x))
    beta = demo_fit(x, y, w, i, delta=delta)
    yest = beta[0] + beta[1] * x[i]
    residuals = y - yest
    s = np.median(np.abs(residuals))
    if s < 1e-3:
        return np.ones_like(x)
    # delta = np.clip(residuals / (6.0 * s), -1.0, 1.0)
    delta = np.minimum(1.0, np.maximum(residuals / (6.0 * s), -1.0))
    delta = (1 - delta**2) ** 2
    return delta
