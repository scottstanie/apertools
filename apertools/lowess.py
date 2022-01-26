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


def lowess_stack(stack, x, frac=2.0 / 3.0, min_x_weighted=None, n_iter=2, n_jobs=-1):
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
def lowess_pixel(y, x, frac=2.0 / 3.0, n_iter=2):
    """Lowess smoother requiring native endian datatype (for numba).
    Performs multiple iterations for robust fitting, downweighting
    by the residuals of the previous fit."""
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
    return xr.DataArray(out_stack, coords=da.coords, dims=da.dims)


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


def demo_window(x, frac=2.0 / 3.0, min_x_weighted=None):
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
