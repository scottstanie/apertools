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
import numpy as np
from numba import njit
from matplotlib.dates import date2num
import pymp


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
    # Make each depth-wise pixel into a column (so there's only 1 OMP loop)
    stack_cols = stack.reshape((ns, -1))

    if min_x_weighted:
        frac = _find_frac(x, min_x_weighted)

    # stack_out = np.zeros(stack_cols.shape, dtype=stack.dtype)
    # for index in range(0, stack_cols.shape[1]):
    stack_out = pymp.shared.array(stack_cols.shape, dtype=stack.dtype)
    with pymp.Parallel(n_jobs) as p:
        for index in p.range(0, stack_cols.shape[1]):
            y = stack_cols[:, index]
            stack_out[:, index] = lowess_pixel(y, x, frac, n_iter)

    return stack_out.reshape((ns, nrows, ncols))


@njit(cache=True, nogil=True)
def lowess_pixel(y, x, frac=0.4, n_iter=2):
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

    Returns:
        ndarray: smoothed y values with shape (ns,)
    """
    n = len(x)
    yest = np.zeros(n)
    if np.any(np.isnan(y)) or np.all(y == 0.0):
        return yest

    r = int(np.ceil(frac * n))
    if r == n:
        r -= 1

    # Find the distance to the rth furthest point from each x value
    h = np.array([np.sort(np.abs(x - x[i]))[r] for i in range(n)])
    xc = x.copy()  # make contiguous (necessary for `reshape` for numba)
    # Get the relative distance to the rth furthest point
    w = np.abs((xc.reshape((-1, 1)) - xc.reshape((1, -1))) / h)
    # Clip to 0, 1 (`np.clip` not available in numba)
    w = np.minimum(1.0, np.maximum(w, 0.0))
    # tricube weighting
    w = (1 - w ** 3) ** 3

    # Initialize the residual-weights to 1
    delta = np.ones(n)

    for _ in range(n_iter):
        for i in range(n):
            weights = delta * w[:, i]
            # Form the linear system as the reduced-size version of the Ax=b:
            # A^T A x = A^T b
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array(
                [
                    [np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)],
                ]
            )

            beta = np.linalg.lstsq(A, b)[0]
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        if s < 1e-3:
            break
        # delta = np.clip(residuals / (6.0 * s), -1.0, 1.0)
        delta = np.minimum(1.0, np.maximum(residuals / (6.0 * s), -1.0))
        delta = (1 - delta ** 2) ** 2

    return yest

    # """Run lowess on a DataArray stack"""


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
        frac = _find_frac(x, min_days_weighted)
    out_stack = lowess_stack(da.values, x, frac, n_iter)
    # Now return as a DataArray
    out_da = xr.DataArray(out_stack, coords=da.coords, dims=da.dims)
    out_da.attrs["description"] = "Bootstrap mean of lowess smoothed stack"

    out_da = _write_attrs(out_da, frac=frac, n_iter=n_iter)


def _find_frac(x, min_x_weighted):
    """Find fraction of data to use so all windows are at least `min_x_weighted` days

    Args:
        x (ndarray): array of days from date2num
        min_x_weighted (int, float): Minimum time period (in units of `x`) to include in lowess fit
    """
    n = len(x)
    day_diffs = np.abs(x.reshape((-1, 1)) - x.reshape((1, -1)))
    # The `kk`th diagonal will contain the time differences when using points `kk` indices apart
    # (e.g. the 2nd diagonal contains how many days apart are x[2]-x[0], x[3]-x[1],...)
    smallest_diffs = np.array([np.min(np.diag(day_diffs, k=kk)) for kk in range(n)])
    # Get the first diagonal where it crosses the min_x_weighted threshold
    idxs_larger = np.where(smallest_diffs > min_x_weighted)[0]
    if len(idxs_larger) > 0:
        return idxs_larger[0] / n
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


def demo_window(x, frac=0.4, min_x_weighted=None):
    if min_x_weighted:
        frac = _find_frac(x, min_x_weighted)
    n = len(x)
    r = int(np.ceil(frac * n))
    if r == n:
        r -= 1

    # Find the distance to the rth furthest point from each x value
    h = np.array([np.sort(np.abs(x - x[i]))[r] for i in range(n)])
    xc = x.copy()  # make contiguous (necessary for `reshape` for numba)
    # Get the relative distance to the rth furthest point
    w = np.abs((xc.reshape((-1, 1)) - xc.reshape((1, -1))) / h)
    # Clip to 0, 1 (`np.clip` not available in numba)
    w = np.minimum(1.0, np.maximum(w, 0.0))
    # tricube weighting
    w = (1 - w ** 3) ** 3
    return w


@njit(cache=True, nogil=True)
def bootstrap_mean_std(x, y, frac=0.3, n_iter=2, K=100, pct_bootstrap=1.0):
    """Bootstrap mean and standard deviation of y

    Args:
        x (ndarray): x values
        y (ndarray): y values
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
        x,
        y,
        frac=frac,
        n_iter=n_iter,
        K=K,
        pct_bootstrap=pct_bootstrap,
    )
    mean = np_nanmean(bootstraps, axis=0)
    std = np_nanstd(bootstraps, axis=0)
    return mean, std


@njit(cache=True, nogil=True)
def bootstrap_lowess(x, y, frac=0.3, n_iter=2, K=100, pct_bootstrap=1.0):
    """Repeatedly run lowess on a pixel, bootstrapping the data

    Args:
        x (ndarray): x values
        y (ndarray): y values
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

    fig, ax = plt.subplots()
    ax.fill_between(xplot, mean - 1.96 * std, mean + 1.96 * std, alpha=0.25)
    ax.plot(xplot, mean, color="red")
    if y is not None:
        ax.plot(xplot, y, ".-")
    ax.set_title(f"{frac = :.2f}, {K = }, {pct_bootstrap = :.2f}")
    return fig, ax


def demo_fit(x, y, w, i, delta=None):
    if delta is None:
        delta = np.ones(len(x))
    weights = delta * w[:, i]
    # Form the linear system as the reduced-size version of the Ax=b:
    # A^T A x = A^T b
    b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
    A = np.array(
        [
            [np.sum(weights), np.sum(weights * x)],
            [np.sum(weights * x), np.sum(weights * x * x)],
        ]
    )

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
    delta = (1 - delta ** 2) ** 2
    return delta


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
def bootstrap_lowess_xr(
    da,
    x_dset="date",
    min_days_weighted=2 * 365.25,
    frac=0.4,
    n_iter=2,
    n_jobs=-1,
    K=100,
    pct_bootstrap=1.0,
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
        frac = _find_frac(x, min_days_weighted)

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
        frac = _find_frac(x, min_days_weighted)

    # stack_out = np.zeros(stack_cols.shape, dtype=stack.dtype)
    # for index in range(0, stack_cols.shape[1]):
    stack_out_mean = pymp.shared.array(stack_cols.shape, dtype=stack.dtype)
    stack_out_std = pymp.shared.array(stack_cols.shape, dtype=stack.dtype)
    with pymp.Parallel(n_jobs) as p:
        for index in p.range(0, stack_cols.shape[1]):
            y = stack_cols[:, index]
            mean, std = bootstrap_mean_std(
                x, y, frac=frac, n_iter=n_iter, K=K, pct_bootstrap=pct_bootstrap
            )
            stack_out_mean[:, index] = mean
            stack_out_std[:, index] = std

    # return stack_out.reshape((ns, nrows, ncols))
    # Now return as a DataArray
    out_mean = xr.DataArray(stack_out_mean, coords=da.coords, dims=da.dims)
    out_std = xr.DataArray(stack_out_std, coords=da.coords, dims=da.dims)
    out_mean.attrs["description"] = "Bootstrap mean of lowess smoothed stack"
    out_std.attrs[
        "description"
    ] = "Bootstrap standard deviation of lowess smoothed stack"

    out_mean = _write_attrs(
        out_mean, K=K, frac=frac, n_iter=n_iter, pct_bootstrap=pct_bootstrap
    )
    out_std = _write_attrs(
        out_std, K=K, frac=frac, n_iter=n_iter, pct_bootstrap=pct_bootstrap
    )
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
