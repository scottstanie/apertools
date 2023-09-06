import datetime
import os
import re
import subprocess
from glob import glob
from pathlib import Path
from pprint import pprint

import numpy as np
import rasterio as rio  # TODO: figure out if i want rio or gdal...
from rasterio.vrt import WarpedVRT
from shapely import geometry
from tqdm import tqdm

from apertools import geojson
from apertools.deramp import remove_ramp
from apertools.latlon import km_to_deg
from apertools.log import get_log
from apertools.utils import chdir_then_revert

logger = get_log()

COMP_DICT_BLOSC = {"compression": 32001, "compression_opts": (0, 0, 0, 0, 5, 1, 1)}
COMP_DICT_LZF = {"compressions": "LZF"}
CRS_LATLON = "EPSG:4326"


def read_intersections(
    fname1=None,
    fname2=None,
    band1=None,
    band2=None,
    nodata=0,
    mask_intersection=True,
    target_crs=None,
    verbose=False,
    res=None,
):
    """Read in the intersection of 2 files as an array"""
    bounds = get_intersection_bounds(fname1, fname2, target_crs=target_crs)
    if verbose:
        logger.info(f"bounds: {bounds}")
    im1 = copy_vrt(fname1, out_fname="", bbox=bounds, band=band1, dst_crs=target_crs, res=res)
    im2 = copy_vrt(fname2, out_fname="", bbox=bounds, band=band2, dst_crs=target_crs, res=res)
    if mask_intersection:
        nodata1 = get_nodata(fname1)
        nodata2 = get_nodata(fname2)
        for nd in [nodata, nodata1, nodata2]:
            cur_mask = np.logical_or(im1 == nd, im2 == nd)
            im1[cur_mask] = nodata
            im2[cur_mask] = nodata
    return im1, im2


# def read_intersections_xr(ds1, ds2, xdim="lon", ydim="lat"):
# bounds = get_intersection_bounds(ds1=ds1, ds2=ds2, xdim=xdim, ydim=ydim)


def read_unions(fname1, fname2, band1=None, band2=None):
    """Read in the union of 2 files as an array"""
    bounds = get_union_bounds(fname1, fname2)
    # print(f"bounds: {bounds}")
    im1 = copy_vrt(fname1, out_fname="", bbox=bounds, band=band1)
    im2 = copy_vrt(fname2, out_fname="", bbox=bounds, band=band2)
    return im1, im2


def write_intersections(
    fname1, fname2, outname1, outname2, driver=None, nodata=0, mask_intersection=True
):
    """Write the intersection of 2 files to out_fname"""

    im1, im2 = read_intersections(
        fname1, fname2, nodata=nodata, mask_intersection=mask_intersection
    )
    transform = get_intersect_transform(fname1, fname2)
    crs = get_crs(fname1)
    if crs != get_crs(fname2):
        raise ValueError(f"CRS mismatch: {crs} != {get_crs(fname2)}")
    write_outfile(
        outname1, im1, crs=crs, transform=transform, driver=driver, nodata=nodata
    )
    write_outfile(
        outname2, im2, crs=crs, transform=transform, driver=driver, nodata=nodata
    )


# TODO: this is basically the same as copy_subset... def merge these
def copy_vrt(
    in_fname,
    out_fname="",
    bbox=None,
    verbose=False,
    band=None,
    dst_crs=CRS_LATLON,
    res=None,
):
    """Create a VRT for (a subset of) a gdal-readable file

    bbox format: (left, bottom, right, top)"""
    from osgeo import gdal

    # if not out_fname:
    # out_fname = in_fname + ".vrt"
    # Using Translate... but would use Warp if every reprojecting
    if bbox:
        left, bottom, right, top = bbox
        projwin = (left, top, right, bottom)  # unclear why Translate does UL LR
    else:
        projwin = None

    if verbose:
        msg = ""
        if bbox:
            msg += f"Subsetting bbox: {bbox} "
        if out_fname:
            msg += f"writing to {out_fname}"
        logger.info(msg)
    if not out_fname:
        out_fname = ""

    # projwin = (left, top, right, bottom)  # unclear why Translate does UL LR
    # out_ds = gdal.Translate(out_fname, in_fname, projWin=projwin)
    # ds_in = gdal.Open(in_fname)
    out_ds = gdal.Warp(
        str(out_fname),
        str(in_fname),
        outputBounds=bbox,
        format="VRT",
        dstSRS=dst_crs,
        xRes=res,
        yRes=res,
    )
    out_arr = out_ds.ReadAsArray()
    if band and out_arr.ndim == 3:
        out_arr = out_arr[band - 1]
    # ds_in, out_ds = None, None
    return out_arr


def read_subset(bbox, in_fname, driver=None, bands=None):
    # This has a bug right now due to rasterio rounding
    # https://github.com/mapbox/rasterio/issues/2004
    # bbox: left, bot, right, top
    with rio.open(in_fname, driver=driver) as src:
        win = src.window(*bbox) if bbox else None
        # TODO: do i want to make this 2d if one band requested?
        return src.read(bands, window=win)


def copy_subset(
    in_fname,
    out_fname,
    bbox=None,
    bounding_file=None,
    driver=None,
    bands=None,
    nodata=0,
    verbose=True,
):
    """Copy a box from `in_fname` to `out_fname`

    Can with specify (left, bot, right, top) with `bbox`, or use the bounds
    of an existing other file in `bounding_file`
    """
    if bbox is None:
        if bounding_file is not None:
            bbox = get_bounds(bounding_file)
    if verbose:
        logger.info(f"Subsetting {bbox} from {in_fname} to {out_fname}")

    img = read_subset(bbox, in_fname, driver)
    crs = get_crs(in_fname, driver=driver)
    transform = get_transform(in_fname, driver=driver, bbox=bbox)

    write_outfile(
        out_fname, img, crs=crs, transform=transform, driver=driver, nodata=nodata
    )


def write_outfile(
    out_fname,
    img,
    mode="w",
    crs=None,
    transform=None,
    driver=None,
    nodata=None,
    dtype=None,
    unit=None,
    **kwargs,
):
    if driver is None and Path(out_fname).suffix == ".tif":
        driver = "GTiff"
    # TODO: do i wanna guess for any others?
    if driver is None:
        raise ValueError("'driver' is required to write dataset")

    if img.ndim < 3:
        img = img[np.newaxis, :, :]
    count = img.shape[0]
    dtype = dtype or img.dtype

    with rio.open(
        out_fname,
        mode,
        count=count,
        crs=crs,
        transform=transform,
        driver=driver,
        height=img.shape[1],
        width=img.shape[2],
        nodata=nodata,
        dtype=dtype,
    ) as dst:
        for band, layer in enumerate(img, start=1):
            dst.write(layer, band)
            if unit:
                dst.set_band_unit(band, unit)


def _get_img_bounds(
    fname1=None,
    fname2=None,
    ds1=None,
    ds2=None,
    xdim="lon",
    ydim="lat",
    target_crs=CRS_LATLON,
):
    """Read 2 arrays and make sure they are on the same projection"""
    if fname1 is not None and fname2 is not None:
        with rio.open(fname1) as src1, rio.open(fname2) as src2:
            if target_crs is not None:
                with WarpedVRT(src1, crs=target_crs) as wsrc2, WarpedVRT(
                    src2, crs=target_crs
                ) as wsrc1:
                    b1 = geometry.box(*wsrc1.bounds)
                    b2 = geometry.box(*wsrc2.bounds)
            else:
                if src1.crs != src2.crs:
                    raise ValueError(
                        f"{fname1} has CRS {src1.crs}, but {fname2} has CRS {src2.crs}"
                    )
                b1 = geometry.box(*src1.bounds)
                b2 = geometry.box(*src2.bounds)
            return b1, b2
    elif ds1 is not None and ds2 is not None:
        s1 = ds1.rio.set_spatial_dims(xdim, ydim)
        s2 = ds2.rio.set_spatial_dims(xdim, ydim)
        b1 = geometry.box(*s1.rio.bounds())
        b2 = geometry.box(*s2.rio.bounds())
        return b1, b2
    else:
        raise ValueError("Must provide either fname or ds")


def get_bounds(fname=None, ds=None, xdim="lon", ydim="lat"):
    """Find the (left, bot, right, top) bounds 1 file `fname`"""
    if fname is not None:
        with rio.open(fname) as src:
            return src.bounds
    elif ds is not None:
        return ds.rio.set_spatial_dims(xdim, ydim).rio.bounds()


def get_intersection_bounds(
    fname1=None,
    fname2=None,
    ds1=None,
    ds2=None,
    xdim="lon",
    ydim="lat",
    target_crs=None,
):
    """Find the (left, bot, right, top) bounds of intersection of fname1 and fname2
    or 2 xarray.Datasets ds1 and ds1"""
    b1, b2 = _get_img_bounds(
        fname1, fname2, ds1, ds2, xdim, ydim, target_crs=target_crs
    )
    return b1.intersection(b2).bounds


def get_union_bounds(
    fname1=None,
    fname2=None,
    ds1=None,
    ds2=None,
    xdim="lon",
    ydim="lat",
    target_crs=None,
):
    """Find the (left, bot, right, top) bounds of union of fname1 and fname2"""
    b1, b2 = _get_img_bounds(
        fname1, fname2, ds1, ds2, xdim, ydim, target_crs=target_crs
    )
    return b1.union(b2).bounds


def get_transform(in_fname, driver=None, bbox=None, target_crs=None):
    with rio.open(in_fname, driver=driver) as src:
        if target_crs is None:
            s = src
        else:
            s = WarpedVRT(src, crs=target_crs)
        transform = s.window_transform(s.window(*bbox)) if bbox else s.transform
        return transform


def get_intersect_transform(fname1, fname2, target_crs=None):
    bbox = get_intersection_bounds(fname1, fname2, target_crs=target_crs)
    return get_transform(fname1, bbox=bbox, target_crs=target_crs)


def get_union_transform(fname1, fname2, target_crs=None):
    bbox = get_union_bounds(fname1, fname2, target_crs=target_crs)
    return get_transform(fname1, bbox=bbox, target_crs=target_crs)


def get_crs(fname, driver=None):
    with rio.open(fname, driver=driver) as src:
        return src.crs


def get_driver(fname):
    with rio.open(fname) as src:
        return src.driver


def get_nodata(fname):
    with rio.open(fname) as src:
        return src.nodata


def bbox_around_point(lons, lats, side_km=25):
    """Finds (left, bot, right top) in deg around arrays of lons, lats

    Returns (N, 4) array, where N = len(lons)"""
    side_deg = km_to_deg(side_km)
    buff = side_deg / 2
    return np.array(
        [
            (lon - buff, lat - buff, lon + buff, lat + buff)
            for (lon, lat) in zip(lons, lats)
        ]
    )


def create_merged_files(
    fname_left,
    fname_right,
    deramp_left=False,
    deramp_right=False,
    deramp_order=1,
    band_left=None,
    band_right=None,
    outfile=None,
    nodata=0,
    unit="centimeters",
    blend=True,
):
    """Create a merged version of two files, bounded by their union bounds"""

    img_left, img_right = read_unions(
        fname_left, fname_right, band1=band_left, band2=band_right
    )
    if deramp_left:
        mask1 = img_left == 0
        img_left = np.nan_to_num(
            remove_ramp(img_left, deramp_order=deramp_order, mask=mask1)
        )
    if deramp_right:
        mask2 = img_right == 0
        img_right = np.nan_to_num(
            remove_ramp(img_right, deramp_order=deramp_order, mask=mask2)
        )

    if blend:
        img_left = np.nan_to_num(img_left, nan=0.0)
        img_right = np.nan_to_num(img_right, nan=0.0)
        mask_left = img_left == 0
        mask_right = img_right == 0
        # mask_tot = np.logical_and(mask_left, mask_right)

        merged = np.zeros_like(img_left)
        for idx in range(img_left.shape[0]):
            ramp_left = np.zeros(img_left.shape[1])
            good_idxs_left = np.where(~mask_left[idx])[0]
            if len(good_idxs_left) < 2:
                continue
            start, end = good_idxs_left[[0, -1]]
            ramp_left[start:end] = np.linspace(1, 0, end - start)
            # ramp_left = np.linspace(1, 0, img_left.shape[1])
            ramp_left[mask_left[idx]] = 0.0

            ramp_right = np.zeros(img_right.shape[1])
            good_idxs_right = np.where(~mask_right[idx])[0]
            if len(good_idxs_right) < 2:
                continue
            start, end = good_idxs_right[[0, -1]]
            ramp_right[start:end] = np.linspace(0, 1, end - start)
            # ramp_right = np.linspace(1, 0, img_left.shape[1])
            ramp_right[mask_right[idx]] = 0.0

            ramp_norm = ramp_left + ramp_right
            # Where there's no data, the sum will be 0. Set to 1 to keep image values
            ramp_norm[ramp_norm == 0] = 1.0
            ramp_left /= ramp_norm
            ramp_right /= ramp_norm

            merged[idx] = img_left[idx] * ramp_left + img_right[idx] * ramp_right
    else:
        valid1 = (img_left != 0).astype(int)
        valid2 = (img_right != 0).astype(int)
        valid_count = valid1 + valid2
        # Where there's no data, the sum will be 0. Set to 1 to keep image values
        valid_count[valid_count == 0] = 1.0
        merged = (img_left + img_right) / valid_count

    if outfile:
        transform = get_union_transform(fname_left, fname_right)
        crs = get_crs(fname_left)
        if crs != get_crs(fname_right):
            raise ValueError(f"CRS mismatch: {crs} != {get_crs(fname_right)}")
        logger.info(f"Writing merged file to {outfile}")
        write_outfile(
            outfile, merged, crs=crs, transform=transform, nodata=nodata, unit=unit
        )
    return merged


# ######## Project Wide subset functions (ISCE) ##############
ISCE_STRIPMAP_PROJECT_FILES = [
    # (directory name, search string for glob())
    ("geom_reference/", "*.rdr"),
    ("merged/SLC/", "**/*.slc"),
    ("Igrams/", "**/*.int"),
    ("Igrams/", "**/*.cor"),
    ("Igrams/", "**/*.unw"),
]


def crop_isce_project(
    bbox_rdr=None,
    bbox_latlon=None,
    project_dir=".",
    output_dir="cropped",
    verbose=True,
    overwrite=False,
    project_files=ISCE_STRIPMAP_PROJECT_FILES,
):
    import isce  # noqa
    import isceobj

    if bbox_rdr is None and bbox_latlon is None:
        raise ValueError("need either bbox_rdr or bbox_latlon")
    elif bbox_latlon:
        from apertools import latlon

        lon0, lat0, lon1, lat1 = bbox_latlon
        geom_dir = os.path.join(project_dir, "geom_reference")
        azbot, rgmin = latlon.latlon_to_rowcol_rdr(lat0, lon0, geom_dir=geom_dir)
        aztop, rgmax = latlon.latlon_to_rowcol_rdr(lat1, lon1, geom_dir=geom_dir)
        bbox_rdr = rgmin, azbot, rgmax, aztop
    # else:
    # rgmin, azbot, rgmax, aztop = bbox_rdr
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(output_dir, "bbox_rdr.wkt"), "w") as f:
        f.write(geojson.bbox_to_wkt(bbox_rdr) + "\n")

    # Get the affine tranform the the multilooked version. use for all (including SLC)
    cmd_base = "gdal_translate -projwin {ulx} {uly} {lrx} {lry} -of ISCE {inp} {out}"
    transform = None
    failures = []
    ifg_folder_pat = re.compile(r"(?P<folder>\d{8}_\d{8})")
    slc_folder_pat = re.compile(r"(?P<folder>\d{8})")
    for dirname, fileglob in tqdm(project_files, position=0):
        filelist = glob(os.path.join(project_dir, dirname, fileglob))
        tqdm.write(
            "Found %s files to subset for %s/%s" % (len(filelist), dirname, fileglob)
        )
        Path(os.path.join(output_dir, dirname)).mkdir(exist_ok=True, parents=True)
        for f in tqdm(filelist, position=1):
            if transform is None:
                with rio.open(f) as src:
                    transform = src.transform

            _, filename = os.path.split(f)
            # For SLCs or ifgs, add back in the sub-folder with the SAR date/ifg dates
            if "slc" in f.lower() or "igrams" in f.lower():
                pat = slc_folder_pat if "slc" in f.lower() else ifg_folder_pat
                match = re.search(pat, f)
                subfolder = match.group()
                newdir = os.path.join(output_dir, dirname, subfolder)
                Path(newdir).mkdir(exist_ok=True, parents=True)
            else:
                newdir = os.path.join(output_dir, dirname)
            outname = os.path.join(newdir, filename)

            if os.path.exists(outname) and not overwrite:
                if verbose:
                    tqdm.write("%s exists, skipping" % outname)
                continue

            if verbose:
                tqdm.write("Subsetting %s to %s" % (f, outname))

            ulx, uly, lrx, lry = get_bounds_rdr(bbox_rdr, fname=f, transform=transform)
            cmd = cmd_base.format(
                inp=f,
                out=outname,
                ulx=ulx,
                uly=uly,
                lrx=lrx,
                lry=lry,
            )
            if verbose:
                tqdm.write(cmd)

            try:
                subprocess.check_output(cmd, shell=True)
                with chdir_then_revert(newdir):
                    # Also add the .vrt header, which doesn't seem to appear from gdal
                    img = isceobj.createImage()
                    img.load(filename + ".xml")
                    img.renderHdr()
            except subprocess.CalledProcessError:
                tqdm.write("Command dailed on %s:" % f)
                tqdm.write(cmd)
                failures.append((f, outname, cmd))
                continue

    logger.info("Failed on the following:")
    pprint(failures)
    return failures


def get_bounds_rdr(bbox_rdr, fname=None, transform=None):
    """Convert the multilooked radar coords, using the Affine transform, to full sized coords

    https://github.com/sgillies/affine
    """
    if transform is None:
        with rio.open(fname) as src:
            transform = src.transform

    left, bot, right, top = bbox_rdr
    ulx, uly = transform * (left, top)
    lrx, lry = transform * (right, bot)
    return ulx, uly, lrx, lry


GEOCODED_PROJECT_FILES = [
    "unw_stack.h5",
    "cor_stack.h5",
    "ifg_stack.h5",
    "masks.h5",
    "elevation_looked.dem",
    "los_enu.tif",
]

COMP_DSETS = {
    "unw_stack.h5": ["stack_flat_shifted"],
    "cor_stack.h5": ["stack"],
    "masks.h5": ["ifg", "slc"],
    "ifg_stack.h5": ["stack"],
}


def crop_geocoded_project(
    bbox=None,
    bbox_file=None,
    project_dir=".",
    files_to_crop=GEOCODED_PROJECT_FILES,
    dsets_to_compress=COMP_DSETS,
    output_dir="cropped",
):
    """Subset all important data stacks for geocoded project by bounding box

    Args:
        bbox (Iterable[float]): (left, bot, right, top) bounding box.
        bbox_file (str, optional): file containing bounding box, either wkt or geojson.
        project_dir (str): path to the original, full HDF5 dataset
        fname (str, optional): filename of the HDF5 file at `full_path`. Defaults to "unw_stack.h5".
        output_dir (str): path to save the output subsetted dataset
    """
    import rioxarray

    if bbox_file:
        bbox = geojson.load_bbox(bbox_file)

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    with open(os.path.join(output_dir, "bbox.geojson"), "w") as f:
        f.write(geojson.bbox_to_geojson(bbox) + "\n")
    with open(os.path.join(output_dir, "bbox.wkt"), "w") as f:
        f.write(geojson.bbox_to_wkt(bbox) + "\n")

    for fname in files_to_crop:
        filepath = os.path.join(project_dir, fname)
        logger.info("Subsetting %s", filepath)
        if not os.path.exists(filepath):
            logger.warning("%s doesn't exists. Skipping subset", filepath)
            continue

        if fname.endswith(".h5"):
            dsets = dsets_to_compress[fname]
            subset_h5(
                project_dir,
                output_dir,
                fname=fname,
                bbox=bbox,
                dsets_to_compress=dsets,
            )
        else:
            f = os.path.split(fname)[1]
            outname = os.path.join(output_dir, f)
            if os.path.exists(outname):
                logger.info("%s exists, skipping", outname)

            filepath = os.path.join(project_dir, fname)
            ds = rioxarray.open_rasterio(filepath)
            # rioxarray uses x/y instead of lat/lon
            ds_sub = ds.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1]))
            driver = "ROI_PAC" if outname.endswith(".dem") else None
            ds_sub.rio.to_raster(outname, driver=driver)


def subset_h5(
    full_path,
    output_dir,
    bbox,
    fname="unw_stack.h5",
    dsets_to_compress=[],
    save_as_nc=False,
):
    """Save a subset of a HDF5 lat/lon dataset by bounding box

    Args:
        full_path (str): path to the original, full HDF5 dataset
        output_dir (str): path to save the output subsetted dataset
        bbox (Iterable[float]): (left, bot, right, top) bounding box.
        fname (str, optional): filename of the HDF5 file at `full_path`. Defaults to "unw_stack.h5".
        dsets_to_compress (list[str], optional): within `fname`, names of datasets to transfer.
        save_as_nc (bool): Save output with ".nc" netcdf extensions. If False, saves as .h5
    """

    import xarray as xr

    if save_as_nc:
        ext = os.path.splitext(fname)[1]
        f_in = os.path.join(output_dir, fname)
        f_out = f_in.replace(ext, ".nc")
    else:
        f_out = os.path.join(output_dir, fname)

    if os.path.exists(f_out):
        logger.info("%s exists, skipping", f_out)
        return

    ds = xr.open_dataset(
        os.path.join(full_path, fname), engine="h5netcdf", phony_dims="sort"
    )
    compressions = {ds: COMP_DICT_BLOSC for ds in dsets_to_compress}
    ds_sub = ds.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[3], bbox[1]))
    logger.info("Input dimensions: %s", ds.dims)
    logger.info("Output dimensions: %s", ds_sub.dims)
    logger.info("Writing to %s, compressing with: %s", f_out, compressions)
    ds_sub.to_netcdf(f_out, engine="h5netcdf", encoding=compressions, mode="a")


def crop_stacks_by_date(
    min_date=None,
    max_date=None,
    max_temporal_baseline=None,
    max_temporal_bandwidth=None,
    bbox=None,
    project_dir=".",
    out_dir=".",
    files_to_crop=list(COMP_DSETS.keys()),
):
    """Subset all important data stacks for geocoded project by bounding box

    Args:
        bbox (Iterable[float]): (left, bot, right, top) bounding box.
        bbox_file (str, optional): file containing bounding box, either wkt or geojson.
        project_dir (str): path to the original, full HDF5 dataset
        fname (str, optional): filename of the HDF5 file at `full_path`. Defaults to "unw_stack.h5".
        output_dir (str): path to save the output subsetted dataset
    """
    import xarray as xr

    from apertools import sario, utils

    fmt = "%Y%m%d"
    out_str = ""
    if max_temporal_baseline:
        out_str += "_maxtemp{}".format(max_temporal_baseline)
    if max_temporal_bandwidth:
        out_str += "_maxbw{}".format(max_temporal_bandwidth)

    if min_date and max_date:
        out_str += "_{}_{}".format(min_date.strftime(fmt), max_date.strftime(fmt))
    elif min_date:
        out_str += "_{}".format(min_date.strftime(fmt))
    elif min_date:
        out_str += "_{}".format(max_date.strftime(fmt))

    if not out_str:
        raise ValueError("No limiting criteria provided.")

    if min_date:
        min_date = utils.to_datetime(min_date)
    else:
        min_date = datetime.datetime(1900, 1, 1, tzinfo=datetime.timezone.utc)
    if max_date:
        max_date = utils.to_datetime(max_date)
    else:
        max_date = datetime.datetime(2100, 1, 1, tzinfo=datetime.timezone.utc)

    input_files = [os.path.join(project_dir, f) for f in files_to_crop]
    ifglist = sario.load_ifglist_from_h5(input_files[0])
    # dc2 = datetime.datetime(2019, 1, 1, tzinfo=datetime.timezone.utc)
    # ifglist2019 = [
    #     ifg
    #     for ifg in ifglist
    #     if min_date <= ifg[0] <= max_date and min_date <= ifg[1] <= max_date
    # ]
    # ifg_idxs = [ifglist.index(ifg) for ifg in ifglist2019]
    slclist, ifglist, ifg_idxs = utils.filter_slclist_ifglist(
        ifg_date_list=ifglist,
        min_date=min_date,
        max_date=max_date,
        max_temporal_baseline=max_temporal_baseline,
        max_bandwidth=max_temporal_bandwidth,
    )

    # slclist = sario.load_slclist_from_h5(input_files[0])
    # slclist2019 = [slc for slc in slclist if min_date <= slc <= max_date]
    # idxs_slc = [slclist.index(ifg) for ifg in slclist2019]

    for fname in input_files:
        if not os.path.exists(fname):
            logger.warning(f"{fname} doesn't exist: skipping")
            continue
        logger.info("Subsetting %s", fname)
        name_only = os.path.split(fname)[-1]
        # filepath = os.path.join(project_dir, fname)
        ext = os.path.splitext(fname)[1]
        f_out = os.path.join(out_dir, name_only.replace(ext, out_str + ext))
        if os.path.exists(f_out):
            logger.info("%s exists: skipping", f_out)
            continue

        ds = xr.open_dataset(fname, engine="h5netcdf", phony_dims="sort")
        # ds_sub = ds.sel(ifg_idx=ifg_idxs, phony_dim_4=idxs_slc)
        ds_sub = ds.sel(ifg_idx=ifg_idxs)
        if bbox is not None:
            ds_sub = ds_sub.sel(
                lon=slice(bbox[0], bbox[2]), lat=slice(bbox[3], bbox[1])
            )
        logger.info("Input dimensions: %s", ds.dims)
        logger.info("Output dimensions: %s", ds_sub.dims)

        ###
        ###
        dsets_to_compress = COMP_DSETS[name_only]
        compressions = {ds: COMP_DICT_BLOSC for ds in dsets_to_compress}
        logger.info("Writing to %s, compressing with: %s", f_out, compressions)
        ds_sub.to_netcdf(f_out, engine="h5netcdf", encoding=compressions, mode="a")


def read_intersections_xr(
    asc_da,
    desc_da,
    # asc_los_da,
    # desc_los_da,
    date=None,
    xdim="lon",
    ydim="lat",
    crs=CRS_LATLON,
    asc_img_fname="tmp_asc.tif",
    desc_img_fname="tmp_desc.tif",
):
    asc_img_fname, desc_img_fname = "tmp_asc.tif", "tmp_desc.tif"
    # asc_da = asc_xr[asc_dset]
    # desc_da = desc_xr[desc_dset]
    if date is not None:
        asc_da = asc_da.sel(date=date, method="nearest")
        desc_da = desc_da.sel(date=date, method="nearest")
    asc_da.rio.set_spatial_dims(xdim, ydim).rio.set_crs(crs).rio.to_raster(
        asc_img_fname
    )
    desc_da.rio.set_spatial_dims(xdim, ydim).rio.set_crs(crs).rio.to_raster(
        desc_img_fname
    )
    asc_img, desc_img = read_intersections(asc_img_fname, desc_img_fname)
    return asc_img, desc_img

    # # Save and read in the LOS stack overlap
    # name_asc2, name_desc2 = "tmp_asc_los.tif", "tmp_desc_los.tif"
    # asc_los_da.rio.set_spatial_dims(xdim, ydim).rio.set_crs(crs).rio.to_raster(name_asc2)
    # desc_los_da.rio.set_spatial_dims(xdim, ydim).rio.set_crs(crs).rio.to_raster(name_desc2)
    # asc_enu_stack, desc_enu_stack = read_intersections(name_asc2, name_desc2)
    # return asc_img,
