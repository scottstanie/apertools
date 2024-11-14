"""Functions to deal with Line of sight vector computation
"""

import os
import numpy as np
from numpy import sin, cos

# from scipy import interpolate
from apertools import subset, deramp
from apertools.log import get_log

# Default for now
DEFO_DSET = "defo_lowess_shifted"

logger = get_log()


def solve_east_up(
    asc_enu_fname=None,
    desc_enu_fname=None,
    asc_img_fname=None,
    desc_img_fname=None,
    asc_band=1,
    desc_band=1,
    asc_xr=None,
    desc_xr=None,
    xdim="lon",
    ydim="lat",
    asc_dset=DEFO_DSET,
    desc_dset=DEFO_DSET,
    date=None,
    los_dset="los_enu",
    nodata=0,
    crs="EPSG:4326",
    res=None,
    outfile=None,
    unit="centimeters",
    asc_shift=0.0,
    desc_shift=0.0,
    deramp_order=0,
    deramp_mask_thresh=3,
):
    if asc_xr is not None:
        # Save and read in the overlapping region of ascending/descending
        asc_img_fname, desc_img_fname = "tmp_asc.tif", "tmp_desc.tif"
        asc_da = asc_xr[asc_dset]
        desc_da = desc_xr[desc_dset]
        if date is not None:
            asc_da = asc_da.sel(date=date, method="nearest")
            desc_da = desc_da.sel(date=date, method="nearest")
        asc_da.rio.set_spatial_dims(xdim, ydim).rio.set_crs(crs).rio.to_raster(
            asc_img_fname
        )
        desc_da.rio.set_spatial_dims(xdim, ydim).rio.set_crs(crs).rio.to_raster(
            desc_img_fname
        )
        asc_img, desc_img = subset.read_intersections(
            asc_img_fname, desc_img_fname, target_crs=crs, res=res
        )

        # Save and read in the LOS stack overlap
        name_asc2, name_desc2 = "tmp_asc_los.tif", "tmp_desc_los.tif"
        asc_xr[los_dset].rio.set_spatial_dims(xdim, ydim).rio.set_crs(
            crs
        ).rio.to_raster(name_asc2)
        desc_xr[los_dset].rio.set_spatial_dims(xdim, ydim).rio.set_crs(
            crs
        ).rio.to_raster(name_desc2)
        asc_enu_stack, desc_enu_stack = subset.read_intersections(name_asc2, name_desc2)

    elif asc_img_fname is not None:
        asc_enu_stack, desc_enu_stack = subset.read_intersections(
            asc_enu_fname,
            desc_enu_fname,
            target_crs=crs,
            nodata=nodata,
            res=res,
        )
        asc_img, desc_img = subset.read_intersections(
            asc_img_fname,
            desc_img_fname,
            asc_band,
            desc_band,
            target_crs=crs,
            nodata=nodata,
            res=res,
        )

    east, up = solve_east_up_imgs(
        asc_img,
        desc_img,
        asc_enu_stack,
        desc_enu_stack,
        nodata=nodata,
        asc_shift=asc_shift,
        desc_shift=desc_shift,
        deramp_order=deramp_order,
        deramp_mask_thresh=deramp_mask_thresh,
    )

    if outfile:
        transform = subset.get_intersect_transform(
            asc_img_fname, desc_img_fname, target_crs=crs
        )
        if crs is None:
            crs = subset.get_crs(asc_img_fname)
        if nodata is not None:
            nodata = subset.get_nodata(asc_img_fname)
        out_stack = np.stack([east, up], axis=0)
        subset.write_outfile(
            outfile,
            out_stack,
            transform=transform,
            crs=crs,
            nodata=nodata,
            unit=unit,
        )
    # Finally remove the temporary files for xarray
    if asc_xr is not None:
        # import xarray as xr
        for f in [asc_img_fname, desc_img_fname, name_asc2, name_desc2]:
            os.remove(f)
        # Need the intersection coords... maybe from using rioxarray
        # east = xr.DataArray(east, coords=asc_xr[asc_dset].coords)
        # up = xr.DataArray(up, coords=asc_xr[asc_dset].coords)
    return east, up


def solve_east_up_imgs(
    asc_img,
    desc_img,
    asc_enu_stack,
    desc_enu_stack,
    asc_shift=0.0,
    desc_shift=0.0,
    deramp_order=0,
    nodata=0,
    deramp_mask_thresh=3,
):
    if np.isnan(nodata):
        mask = np.logical_or(np.isnan(asc_img), np.isnan(desc_img))
    else:
        mask = np.logical_or(asc_img == 0, desc_img == 0)

    if deramp_order:
        asc_img = _deramp(asc_img, deramp_order, deramp_mask_thresh)
        desc_img = _deramp(desc_img, deramp_order, deramp_mask_thresh)

    asc_img += asc_shift
    desc_img += desc_shift

    if asc_img.shape != desc_img.shape:
        raise ValueError("asc_img not same shape as desc_img")
    if asc_enu_stack.shape != desc_enu_stack.shape:
        raise ValueError("asc_enu_stack not same shape as desc_enu_stack")
    # if asc_enu_stack.shape[1:] != asc_img.shape:
    #     raise ValueError("LOS stack not same shape as image")
    # TODO: change this to warp better
    rows, cols = asc_img.shape
    asc_enu_stack = asc_enu_stack[:, :rows, :cols]
    desc_enu_stack = desc_enu_stack[:, :rows, :cols]
    # ###

    # Form a (2, 2, npixels) array of system matrices A
    # each (2,2) is [asc_east  asc_up; desc_east  desc_up]
    asc_eu_vecs = asc_enu_stack.reshape((3, -1))[::2, :]  # just need E,U of ENU
    desc_eu_vecs = desc_enu_stack.reshape((3, -1))[::2, :]
    asc_desc_eu = np.stack((asc_eu_vecs, desc_eu_vecs), axis=0)

    asc_desc_img = np.stack((asc_img, desc_img), axis=0).reshape((2, -1))

    # Input: (..., M, N) stack of matrices to be pseudo-inverted.
    # output: (..., N, M) after pseudo inverse
    Apinv = np.linalg.pinv(np.moveaxis(asc_desc_eu, -1, 0))
    # This einsum results in (npixel, 2), where each row is [east, up]
    east_up_rows = np.einsum("ijk, ki -> ij", Apinv, asc_desc_img)
    east = east_up_rows[:, 0].reshape(asc_img.shape).astype(np.float32)
    up = east_up_rows[:, 1].reshape(asc_img.shape).astype(np.float32)

    east[mask] = nodata
    up[mask] = nodata

    return east, up


def find_enu_coeffs(
    lon,
    lat,
    los_map_file=None,
    los_da=None,
    coordinates=None,
    geom_dir="geom_reference",
):
    """For arbitrary lat/lon, find the coefficients for ENU components of LOS vector

    Args:
        lon (float): longitude of point to get LOS vector
        lat (float): latitude of point
        los_map_file (str): name of 3-band image with E,N,U as bands.
        los_array (xr.DataArray): 3-band DataArray with E,N,U as bands, 'lat', 'lon' coords
        coordinates (str): ['geo', 'rdr']. Pass 'rdr' if `los_map_file` is in
            radar coordinates
        geom_dir (str): if `los_map_file` is in radar coordinates, directory
            containing `lat.rdr` and `lon.rdr` files

    Returns:
        ndarray: enu_coeffs, shape = (3,) array [alpha_e, alpha_n, alpha_up]
        Pointing from satellite to ground
        Can be used to project an ENU vector into the line of sight direction
    """
    import rasterio as rio

    # Prefer to use the DataArray if it's passed in
    if los_da is not None:
        # Check that lat/lon is inbounds:
        if (lat < los_da.lat.min()) or (lat > los_da.lat.max()):
            raise ValueError("lat out of bounds")
        if (lon < los_da.lon.min()) or (lon > los_da.lon.max()):
            raise ValueError("lon out of bounds")
        return los_da.sel(lat=lat, lon=lon, method="nearest").values.ravel()

    if los_map_file is None:
        raise ValueError("los_map_file or los_da is required")

    with rio.open(los_map_file) as src:
        # src.index(-121.915628095, 37.87857542)  # (1250, -566)
        if coordinates == "rdr":
            from apertools import latlon

            row, col = latlon.latlon_to_rowcol_rdr(lat, lon, geom_dir=geom_dir)
        else:
            row, col = src.index(lon, lat)

        nrows, ncols = src.shape[-2:]
        if any([row < 0, col < 0, row >= nrows, col >= ncols]):
            raise ValueError(f"({lon}, {lat}) is out of bounds from {src.bounds}")
        return src.read(window=((row, row + 1), (col, col + 1))).ravel()


def _deramp(img, deramp_order, thresh):
    mask = np.logical_or(img == 0, np.isnan(img))
    maskbig = np.abs(img) > thresh
    m = np.logical_or(mask, maskbig)

    out = deramp.remove_ramp(img, deramp_order=deramp_order, mask=m, copy=True)
    out[mask] = np.nan
    return out


# TODO: fix this for having premade map
def los_to_enu(los_file=None, lat_lons=None, xyz_los_vecs=None):
    """Converts Line of sight vectors from xyz to ENU

    Can read in the LOS vec file, or take a list `xyz_los_vecs`
    Args:
        los_file (str): file to the recorded LOS vector at lat,lon points
        lat_lons (list[tuple[float]]): list of (lat, lon) coordinares for LOS vecs
        xyz_los_vecs (list[tuple[float]]): list of xyz LOS vectors

    Notes:
        Second two args are the result of read_los_output, mutually
        exclusive with los_file

    Returns:
        ndarray: k x 3 ENU 3-vectors
    """
    # if los_file:
    # lat_lons, xyz_los_vecs = read_los_output(los_file)
    return convert_xyz_latlon_to_enu(lat_lons, xyz_los_vecs)


def convert_xyz_latlon_to_enu(lat_lons, xyz_array):
    return np.array([
        rotate_xyz_to_enu(xyz, lat, lon) for (lat, lon), xyz in zip(lat_lons, xyz_array)
    ])


def rotate_xyz_to_enu(xyz, lat, lon):
    """Rotates a vector in XYZ coords to ENU

    Args:
        xyz (list[float], ndarray[float]): length 3 x,y,z coordinates, either
            as list of 3, or a 3xk array of k ENU vectors
        lat (float): latitude (deg) of point to rotate into
        lon (float): longitude (deg) of point to rotate into

    Reference:
    https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates

    """
    # Rotate about axis 3 with longitude, then axis 1 with latitude
    R3 = rot(90 + lon, 3, in_degrees=True)
    R1 = rot(90 - lat, 1, in_degrees=True)
    return np.matmul(R1, np.matmul(R3, xyz))


def rot(angle, axis, in_degrees=True):
    """
    Find a 3x3 euler rotation matrix given an angle and axis.

    Rotation matrix used for rotating a vector about a single axis.

    Args:
        angle (float): angle in degrees to rotate
        axis (int): 1, 2 or 3
        in_degrees (bool): specify the angle in degrees. if false, using
            radians for `angle`
    """
    R = np.eye(3)
    if in_degrees:
        angle = np.deg2rad(angle)
    cang = cos(angle)
    sang = sin(angle)
    if axis == 1:
        R[1, 1] = cang
        R[2, 2] = cang
        R[1, 2] = sang
        R[2, 1] = -sang
    elif axis == 2:
        R[0, 0] = cang
        R[2, 2] = cang
        R[0, 2] = -sang
        R[2, 0] = sang
    elif axis == 3:
        R[0, 0] = cang
        R[1, 1] = cang
        R[1, 0] = -sang
        R[0, 1] = sang
    else:
        raise ValueError("axis must be 1, 2 or 2")
    return R


def project_enu_to_los(
    data_enu,
    enu_coeffs=None,
    los_xyz=None,
    lat=None,
    lon=None,
):
    """Find magnitude of an ENU vector in the LOS direction

    Rotates the line of sight vector to ENU coordinates at
    (lat, lon), then dots with the data_enu data vector

    Args:
        data_enu (ndarray[float]): E,N,U coordinates, either
            as list of 3, or a (3, k) array of k ENU vectors
        enu_coeffs (ndarray) size 3 array of the E,N,U coefficients
            of a line of sight vector.
        los_xyz (ndarray[float]): Optional. length 3 line of sight, in XYZ frame
            Needed if enu_coeffs is not provided
        lat (float): degrees latitude of los point
        lon (float): degrees longitude of los point

    Returns:
        ndarray: magnitudes same length as data_enu input, (k, 1)

    Examples:
    >>> print('%.2f' % project_enu_to_los([1.,2.,3.],enu_coeffs=[1., 0, 0]))
    1.00
    """
    if enu_coeffs is None:
        los_hat = los_xyz / np.linalg.norm(los_xyz)
        enu_coeffs = rotate_xyz_to_enu(los_hat, lat, lon)
    return np.array(enu_coeffs) @ np.array(data_enu)


def merge_slclists(slclist1, slclist2):
    """Take asc and desc slclists, makes one merged

    Gives the overlap indices of the merged list for each smaller

    """
    merged_slclist = np.concatenate((slclist1, slclist2))
    merged_slclist.sort()

    _, indices1, _ = np.intersect1d(merged_slclist, slclist1, return_indices=True)
    _, indices2, _ = np.intersect1d(merged_slclist, slclist2, return_indices=True)
    return merged_slclist, indices1, indices2


def merge_datelist_xr(ds1, ds2, col="date"):
    """Find the outer join of ds1[col] and ds2[col]"""
    # Equivalent to...
    # los.merge_slclists(pd.to_datetime(ds85.date.values).values,
    # pd.to_datetime(ds151.date.values).values)
    import pandas as pd

    d = pd.merge(pd.Series(ds1.indexes[col]), pd.Series(ds2.indexes[col]), how="outer")
    out_series = d.sort_values([col]).reset_index(drop=True)
    return out_series.to_xarray().date


def merge_xr(ds1, ds2, freq="6M", col="date", dset1=DEFO_DSET, dset2=DEFO_DSET):
    """Interpolate 2 datasets to a set of dates

    Args:
        ds1 (xr.Dataset): first dataset
        ds2 (xr.Dataset): seconds dataset
        freq (str, optional): Date frequency (passed to pandas). Defaults to "6M".
        col (str, optional): name of time index in ds1/ds2. Defaults to "date".
        dset1 (str, optional): name of data variable for ds1. Defaults to DEFO_DSET.
        dset2 (str, optional): name of data variable for ds2. Defaults to DEFO_DSET.

    Returns:
        tuple[xr.Dataset]: two Datasets, interpolated to the dates specified
        in the data variable `dset1`, `dset2`
    """
    import pandas as pd

    if freq is not None:
        # Take the later of the start dates
        dmin = max(np.min(ds1[col]), np.min(ds2[col])).to_pandas()
        # Take the earlier of the end dates
        dmax = min(np.max(ds1[col]), np.max(ds2[col])).to_pandas()
        # Round to nearest quarters
        dmin = pd.to_datetime(dmin) + pd.tseries.offsets.QuarterBegin()
        dmax = pd.to_datetime(dmax) + pd.tseries.offsets.QuarterEnd()

        dd = pd.date_range(dmin, dmax, freq=freq)
        # kinda convoluted... maybe there's a simpler way to get back to DataArray
        date_range = dd.to_series().to_xarray().rename(index="date")
    else:
        # Otherwise use all the possible dates, outer joined
        date_range = merge_datelist_xr(ds1, ds2, col=col)
    out1 = ds1[dset1].interp(date=date_range, kwargs={"fill_value": "extrapolate"})
    out2 = ds2[dset2].interp(date=date_range, kwargs={"fill_value": "extrapolate"})

    out1, out2 = out1.to_dataset(), out2.to_dataset()
    return out1, out2
