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

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)
#
# https://gist.github.com/agramfort/850437

import numpy as np
from numba import njit


def lowess(y, x, f=2.0 / 3.0, n_iter=3):
    """Lowess smoother (robust locally weighted regression).

    Fits a nonparametric regression curve to a scatterplot.

    Parameters
    ----------
    y, x : np.ndarrays
        The arrays x and y contain an equal number of elements;
        each pair (x[i], y[i]) defines a data point in the
        scatterplot.

    f : float
        The smoothing span. A larger value will result in a
        smoother curve.
    n_iter : int
        The number of robustifying iteration. Thefunction will
        run faster with a smaller number of iterations.

    Returns
    -------
    yest : np.ndarray
        The estimated (smooth) values of y.

    """
    if not y.dtype.isnative:
        y = y.astype(y.dtype.type)
    return _lowess(y, x, f, n_iter)


@njit(cache=True, nogil=True)
def _lowess(y, x, f=2.0 / 3.0, n_iter=3):  # pragma: no cover
    """Lowess smoother requiring native endian datatype (for numba)."""
    n = len(x)
    r = int(np.ceil(f * n))
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


import pymp

def lowess_stack(stack, x, f=2.0 / 3.0, n_iter=3):
    if not stack.dtype.isnative:
        stack = stack.astype(stack.dtype.type)
    ns, nrows, ncols = stack.shape
    stack_cols = stack.reshape((ns, -1))
    stack_out = pymp.shared.array(stack_cols.shape, dtype=stack.dtype)

    with pymp.Parallel(12) as p:
        for index in p.range(0, stack_cols.shape[1]):
            y = stack_cols[:, index]
            if np.any(np.isnan(y)) or np.all(y == 0):
                continue
            stack_out[:, index] = _lowess(y, x, f, n_iter)
            # p.print('Yay! {} done!'.format(index))
    return stack_out.reshape((ns, nrows, ncols))


import cupy as cp

# @njit(cache=True, nogil=True)
def lowess_cp(yarr, x, f=2.0 / 3.0, n_iter=3):  # pragma: no cover
    """Cupy version of Lowess smoother"""
    xp = cp.get_array_module(x)
    if yarr.ndim == 1:
        # for 1D, k = 1
        yarr = yarr.reshape(-1, 1)
    n = len(x)
    k = yarr.shape[1]

    r = int(xp.ceil(f * n))
    # Find the distance to the rth furthest point from each x value
    h = xp.array([xp.sort(xp.abs(x - x[i]))[r] for i in range(n)])
    # xc = x.copy()  # make contiguous (necessary for `reshape` for numba)
    # Get the relative distance to the rth furthest point
    w = xp.abs((x.reshape((-1, 1)) - x.reshape((1, -1))) / h)
    # Clip to 0, 1 (`clip` not available in numba)
    w = xp.minimum(1.0, xp.maximum(w, 0.0))
    w = (1 - w ** 3) ** 3
    yest = xp.zeros(n, k)
    delta = xp.ones(n)

    for _ in range(n_iter):
        for i in range(n):
            weights = delta[i, :] * w[:, i]
            # b = xp.array([xp.sum(weights * y), xp.sum(weights * y * x)])
            # make b into (2, k) array
            b = np.array(
                [
                    np.sum(weights * yarr, axis=0),
                    np.sum(weights * x[:, None] * yarr, axis=0),
                ]
            )
            A = xp.array(
                [
                    [xp.sum(weights), xp.sum(weights * x)],
                    [xp.sum(weights * x), xp.sum(weights * x * x)],
                ]
            )

            beta = xp.linalg.lstsq(A, b)[0]
            # Evaluate the coeffs for all pixels
            yest[:, i] = [1, x[i]] @ beta
            # yest[i] = beta[0] + beta[1] * x[i]

        residuals = yarr - yest
        s = xp.median(xp.abs(residuals), axis=0)
        assert s.shape == (k,)
        delta = xp.clip(residuals / (6.0 * s), -1.0, 1.0)
        # delta = xp.minimum(1.0, xp.maximum(residuals / (6.0 * s), -1.0))
        delta = (1 - delta ** 2) ** 2

    return yest