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
import pymp


def lowess_stack(stack, x, frac=2.0 / 3.0, n_iter=2, n_jobs=-1):
    """Smooth a stack of images using linear lowess.

    When n_iter > 1, will rerun each regression, reweighting by residuals
    and clipping to 6 MAD standard deviations.

    Args:
        stack (ndarray): stack with shape (ns, nrows, ncols)
        x (ndarray): x values with shape
        frac ([type], optional): fraction of data to use for each smoothing
            Defaults to 2/3
        n_iter (int, optional): Number of iterations to rerun fit after
            reweighting by residuals.
            Defaults to 2.
        n_jobs (int, optional): [description]. Defaults to -1.

    Returns:
        [type]: [description]
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
    stack_cols = stack.reshape((ns, -1))

    stack_out = pymp.shared.array(stack_cols.shape, dtype=stack.dtype)
    with pymp.Parallel(n_jobs) as p:
        for index in p.range(0, stack_cols.shape[1]):
            y = stack_cols[:, index]
            if np.any(np.isnan(y)) or np.all(y == 0):
                continue
            stack_out[:, index] = _lowess(y, x, frac, n_iter)

    return stack_out.reshape((ns, nrows, ncols))


@njit(cache=True, nogil=True)
def _lowess(y, x, frac=2.0 / 3.0, n_iter=2):  # pragma: no cover
    """Lowess smoother requiring native endian datatype (for numba)."""
    n = len(x)
    r = int(np.ceil(frac * n))
    # Find the distance to the rth furthest point from each x value
    h = np.array([np.sort(np.abs(x - x[i]))[r] for i in range(n)])
    xc = x.copy()  # make contiguous (necessary for `reshape` for numba)
    # Get the relative distance to the rth furthest point
    w = np.abs((xc.reshape((-1, 1)) - xc.reshape((1, -1))) / h)
    # Clip to 0, 1 (`clip` not available in numba)
    w = np.minimum(1.0, np.maximum(w, 0.0))
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)

    for _ in range(n_iter):
        for i in range(n):
            weights = delta * w[:, i]
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
        # delta = np.clip(residuals / (6.0 * s), -1.0, 1.0)
        delta = np.minimum(1.0, np.maximum(residuals / (6.0 * s), -1.0))
        delta = (1 - delta ** 2) ** 2

    return yest


def lowess_xr(da, x_dset="date", frac=0.7, n_iter=2):
    """Run lowess on a DataArray stack"""
    from matplotlib.dates import date2num
    import xarray as xr

    x = date2num(da[x_dset].values)
    out_stack = lowess_stack(da.values, x, frac, n_iter)
    # Now return as a DataArray
    return xr.DataArray(out_stack, coords=da.coords, dims=da.dims)

    # TODO: what does  this mean
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
