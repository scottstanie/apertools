import os
import numpy as np
from .log import get_log

logger = get_log()


def remove_ramp(
    z, deramp_order=1, mask=np.ma.nomask, copy=True, dtype=np.float32, dem=None
):
    """Estimates a linear plane through data and subtracts to flatten

    Used to remove noise artifacts from unwrapped interferograms

    Args:
        z (ndarray): 2D array, interpreted as heights
        deramp_order (int): degree of surface estimation
            deramp_order = 1 removes linear ramp, deramp_order = 2 fits quadratic surface

    Returns:
        ndarray: flattened 2D array with estimated surface removed
    """
    if z.ndim > 2:
        if mask.ndim > 2:
            assert len(mask) == len(
                z
            ), f"mask and z must have same length, but {len(mask) = } and {len(z) = }"
        else:
            mask = [mask] * len(z)
        return np.stack(
            [
                remove_ramp(layer, deramp_order, m, copy, dtype)
                for layer, m in zip(z, mask)
            ]
        )

    z_masked = z.copy() if copy else z
    # Make a version of the image with nans in masked places
    z_masked[mask] = np.nan
    # a 0th order ramp is just the constant mean
    # if deramp_order == 0:
    # return (z - np.nanmean(z_masked)).astype(dtype)

    # Use this constrained version to find the plane/quadratic fit
    z_fit = estimate_ramp(z_masked, deramp_order, dem=dem)
    # Then use the non-masked as return value
    return (z - z_fit).astype(dtype)


def estimate_ramp(z, deramp_order, dem=None, save_coeffs=False):
    """Takes a 2D array an fits a linear plane to the data

    Ignores pixels that have nan values

    Args:
        z (ndarray): 2D array, interpreted as heights
        deramp_order (int): degree of surface estimation
            deramp_order = 1 removes linear ramp, deramp_order = 2 fits quadratic surface
        dem (ndarray): 2D array, heights of DEM, same shape as z
            Used remove a linear elevation-phase trend.
            if None, will ignore and not remove trend.
        save_coeffs (bool): Default False. If True, returns the coefficients of the fit
            in addition to the fit.

    Returns:
        ndarray: the estimated polynomial fit for `z`
            If `save_coeffs` is True, returns a tuple of (fit, coeffs)

    Notes:
        for deramp_order = 1, and `dem=None`, fit uses 3 cofficients, a, b, c:
            z = ax + by + c
        For deramp_order = 2, it will be coeffs:
            z = f + ax + by + cxy + dx^2 + ey^2
        where f is the constant term, and y, x are the row and column indices

        When passing `dem`, the fit equation is (e.g., Doin et al. (2009)):
            z = ax + by + c + k*h
            where h is the height of the DEM
    """
    if deramp_order > 2:
        raise ValueError("Order only implemented for 1 and 2")
    # Note: rows == ys, cols are xs
    yidxs, xidxs = matrix_indices(z.shape, flatten=True)
    # c_ stacks 1D arrays as columns into a 2D array
    zflat = z.flatten()
    good_idxs = ~np.isnan(zflat)
    if deramp_order == 0:
        # 0ther order is constant, or only elevation removal
        A = np.c_[np.ones(xidxs.shape)]
        if dem is not None:
            A = np.hstack((A, dem.reshape((-1, 1))))
        coeffs, _, _, _ = np.linalg.lstsq(A[good_idxs], zflat[good_idxs], rcond=None)
        # coeffs will be c in the equation z = c + k*h, or z = c
        c = coeffs[0]
        if dem is None:
            k, dem = 0, 0
        else:
            k = coeffs[1]

        # We want full blocks, as opposed to matrix_index flattened
        y_block, x_block = matrix_indices(z.shape, flatten=False)
        z_fit = c + k * dem

    elif deramp_order == 1:
        A = np.c_[np.ones(xidxs.shape), xidxs, yidxs]
        if dem is not None:
            A = np.hstack((A, dem.reshape((-1, 1))))
        # np.c_[np.ones(xidxs.shape), xidxs, yidxs, dem.flatten()]
        coeffs, _, _, _ = np.linalg.lstsq(A[good_idxs], zflat[good_idxs], rcond=None)
        # coeffs will be a, b, c in the equation z = c + ax + by
        # or z = c + a*x + b*y + k*h
        c, a, b = coeffs[:3]
        if dem is None:
            k, dem = 0, 0
        else:
            k = coeffs[3]

        # We want full blocks, as opposed to matrix_index flattened
        y_block, x_block = matrix_indices(z.shape, flatten=False)
        z_fit = a * x_block + b * y_block + c + k * dem
        # coeffs =

    # TODO: ever want dem + 2nd order? seems like a lot
    elif deramp_order == 2:
        A = np.c_[
            np.ones(xidxs.shape), xidxs, yidxs, xidxs * yidxs, xidxs ** 2, yidxs ** 2
        ]
        # coeffs will be 6 elements for the quadratic
        coeffs, _, _, _ = np.linalg.lstsq(A[good_idxs], zflat[good_idxs], rcond=None)
        yy, xx = matrix_indices(z.shape, flatten=True)
        idx_matrix = np.c_[np.ones(xx.shape), xx, yy, xx * yy, xx ** 2, yy ** 2]
        z_fit = np.dot(idx_matrix, coeffs).reshape(z.shape)

    return z_fit if not save_coeffs else z_fit, coeffs


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


def remove_lowpass(
    z, lowpass_sigma_pct=0.25, mask=np.ma.nomask, copy=True, dtype=np.float32, dem=None
):
    """Subtracts the output of a low-pass filter (aka: performs high pass filter)

    Used to remove noise artifacts from unwrapped interferograms

    Args:
        z (ndarray): 2D array, interpreted as heights
        deramp_order (int): degree of surface estimation
            deramp_order = 1 removes linear ramp, deramp_order = 2 fits quadratic surface

    Returns:
        ndarray: flattened 2D array with estimated surface removed
    """
    import scipy.ndimage as ndi
    from scipy.fft import fft2, ifft2
    from scipy.interpolate import NearestNDInterpolator

    if z.ndim > 2:
        return np.stack(
            [remove_lowpass(layer, lowpass_sigma_pct, mask, copy, dtype) for layer in z]
        )

    z_masked = z.copy() if copy else z
    # For FFT filtering, need 0s instead of nans (or all become nans)
    z_masked[..., mask] = 0

    # z_masked[..., mask] = np.nan
    # nomask = np.where(~mask)
    # interp = NearestNDInterpolator(np.transpose(nomask), z_masked[nomask])
    # z_masked = interp(*np.indices(z_masked.shape))

    # Create the sigma as a percentage of the image size
    sigma = lowpass_sigma_pct * min(z.shape[-2:])

    input_ = fft2(z_masked, workers=-1)
    result = ndi.fourier_gaussian(input_, sigma=sigma)
    z_fit = ifft2(result, workers=-1).real
    z_fit[..., mask] = np.nan

    # Then use the non-masked as return value
    return (z - z_fit).astype(dtype)


def remove_ramp_xr(
    ds,
    dset_name,
    outfile=None,
    deramp_order=1,
    mask=None,
    mask_val=0,
    overwrite=False,
    max_abs_val=None,
):
    from apertools import sario
    if not sario.check_dset(outfile, dset_name, overwrite):
        import xarray as xr
        return xr.open_dataset(outfile)

    logger.info("Removing ramp")

    if mask is None:
        mask = ds[dset_name] == mask_val
    if max_abs_val is not None and max_abs_val > 0:
        mask_abs = np.abs(ds[dset_name]) > max_abs_val
    else:
        mask_abs = np.ma.nomask

    outstack = remove_ramp(
        ds[dset_name].data,
        copy=True,
        deramp_order=deramp_order,
        mask=np.logical_or(mask, mask_abs),
    )
    if mask.ndim == 3:
        outstack[mask] = mask_val
    else:
        outstack[:, mask] = mask_val

    ds_out = ds.copy()
    ds_out[dset_name].data = outstack

    if outfile:
        # ext = os.path.splitext(infile)[1]
        # outfile = infile.replace(ext, "_ramp_removed" + out_format)
        if outfile.endswith("zarr"):
            from numcodecs import Blosc
            compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
            mode = "w" if not os.path.exists(outfile) else "a"
            ds_out.to_zarr(outfile, encoding={"igrams": {"compressor": compressor}}, mode=mode)
        elif outfile.endswith("nc"):
            mode = "w" if not os.path.exists(outfile) else "a"
            ds_out.to_netcdf(outfile, engine="h5netcdf", mode=mode)
    return ds_out
