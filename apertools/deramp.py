import numpy as np


def matrix_indices(shape, flatten=True):
    """Returns a pair of vectors for all indices of a 2D array

    Convenience function to help remembed mgrid syntax

    Example:
        >>> a = np.arange(12).reshape((4, 3))
        >>> print(a)
        [[ 0  1  2]
         [ 3  4  5]
         [ 6  7  8]
         [ 9 10 11]]
        >>> rs, cs = matrix_indices(a.shape)
        >>> rs
        array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        >>> cs
        array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        >>> print(a[rs[1], cs[1]] == a[0, 1])
        True
    """
    nrows, ncols = shape
    row_block, col_block = np.mgrid[0:nrows, 0:ncols]
    if flatten:
        return row_block.flatten(), col_block.flatten()
    else:
        return row_block, col_block


def remove_ramp(z, deramp_order=1, mask=np.ma.nomask, copy=False):
    """Estimates a linear plane through data and subtracts to flatten

    Used to remove noise artifacts from unwrapped interferograms

    Args:
        z (ndarray): 2D array, interpreted as heights
        deramp_order (int): degree of surface estimation
            deramp_order = 1 removes linear ramp, deramp_order = 2 fits quadratic surface

    Returns:
        ndarray: flattened 2D array with estimated surface removed
    """
    z_masked = z.copy() if copy else z
    # Make a version of the image with nans in masked places
    z_masked[mask] = np.nan
    # Use this constrained version to find the plane fit
    z_fit = estimate_ramp(z_masked, deramp_order)
    # Then use the non-masked as return value
    return z - z_fit


def estimate_ramp(z, deramp_order):
    """Takes a 2D array an fits a linear plane to the data

    Ignores pixels that have nan values

    Args:
        z (ndarray): 2D array, interpreted as heights
        deramp_order (int): degree of surface estimation
            deramp_order = 1 removes linear ramp, deramp_order = 2 fits quadratic surface
        deramp_order (int)

    Returns:
        ndarray: the estimated coefficients of the surface
            For deramp_order = 1, it will be 3 numbers, a, b, c from
                 ax + by + c = z
            For deramp_order = 2, it will be 6:
                f + ax + by + cxy + dx^2 + ey^2
    """
    if deramp_order > 2:
        raise ValueError("Order only implemented for 1 and 2")
    # Note: rows == ys, cols are xs
    yidxs, xidxs = matrix_indices(z.shape, flatten=True)
    # c_ stacks 1D arrays as columns into a 2D array
    zflat = z.flatten()
    good_idxs = ~np.isnan(zflat)
    if deramp_order == 1:
        A = np.c_[np.ones(xidxs.shape), xidxs, yidxs]
        coeffs, _, _, _ = np.linalg.lstsq(A[good_idxs], zflat[good_idxs], rcond=None)
        # coeffs will be a, b, c in the equation z = ax + by + c
        c, a, b = coeffs
        # We want full blocks, as opposed to matrix_index flattened
        y_block, x_block = matrix_indices(z.shape, flatten=False)
        z_fit = a * x_block + b * y_block + c

    elif deramp_order == 2:
        A = np.c_[
            np.ones(xidxs.shape), xidxs, yidxs, xidxs * yidxs, xidxs ** 2, yidxs ** 2
        ]
        # coeffs will be 6 elements for the quadratic
        coeffs, _, _, _ = np.linalg.lstsq(A[good_idxs], zflat[good_idxs], rcond=None)
        yy, xx = matrix_indices(z.shape, flatten=True)
        idx_matrix = np.c_[np.ones(xx.shape), xx, yy, xx * yy, xx ** 2, yy ** 2]
        z_fit = np.dot(idx_matrix, coeffs).reshape(z.shape)

    return z_fit
