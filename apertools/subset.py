import os
import re
import datetime
from pprint import pprint
from glob import glob
import subprocess
from tqdm import tqdm
import numpy as np
from shapely import geometry
import rasterio as rio  # TODO: figure out if i want rio or gdal...
from apertools.log import get_log
from apertools.latlon import km_to_deg
from apertools.deramp import remove_ramp
from apertools.utils import mkdir_p, chdir_then_revert
from apertools import geojson

logger = get_log()


# TODO: this is basically the same as copy_subset... def merge these
def copy_vrt(in_fname, out_fname="", bbox=None, verbose=True):
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

    # out_ds = gdal.Translate(out_fname, in_fname, projWin=projwin)
    # ds_in = gdal.Open(in_fname)
    out_ds = gdal.Warp(out_fname, in_fname, outputBounds=bbox, format="VRT")
    out_arr = out_ds.ReadAsArray()
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
):
    if driver is None and out_fname.endswith(".tif"):
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


def _get_img_bounds(fname1, fname2):
    """Read 2 arrays and make sure they are on the same projection"""
    with rio.open(fname1) as src1, rio.open(fname2) as src2:
        if src1.crs != src2.crs:
            raise ValueError(
                f"{fname1} has crs {src1.crs}, but {fname2} has csr {src2.crs}"
            )
        b1 = geometry.box(*src1.bounds)
        b2 = geometry.box(*src2.bounds)
        return b1, b2


def get_intersection_bounds(fname1, fname2):
    """Find the (left, bot, right, top) bounds of intersection of fname1 and fname2"""
    b1, b2 = _get_img_bounds(fname1, fname2)
    return b1.intersection(b2).bounds


def get_union_bounds(fname1, fname2):
    """Find the (left, bot, right, top) bounds of union of fname1 and fname2"""
    b1, b2 = _get_img_bounds(fname1, fname2)
    return b1.union(b2).bounds


def get_transform(in_fname, driver=None, bbox=None):
    with rio.open(in_fname, driver=driver) as src:
        transform = src.window_transform(src.window(*bbox)) if bbox else src.transform
        return transform


def get_bounds(fname):
    """Find the (left, bot, right, top) bounds 1 file `fname`"""
    with rio.open(fname) as src:
        return src.bounds


def get_intersect_transform(fname1, fname2):
    int_bnds = get_intersection_bounds(fname1, fname2)
    return get_transform(fname1, bbox=int_bnds)


def get_crs(fname, driver=None):
    with rio.open(fname, driver=driver) as src:
        return src.crs


def get_driver(fname):
    with rio.open(fname) as src:
        return src.driver


def get_nodata(fname):
    with rio.open(fname) as src:
        return src.nodata


def read_intersections(fname1, fname2, band1=None, band2=None):
    """Read in the intersection of 2 files as an array"""
    bounds = get_intersection_bounds(fname1, fname2)
    print(f"bounds: {bounds}")
    im1 = copy_vrt(fname1, out_fname="", bbox=bounds)
    im2 = copy_vrt(fname2, out_fname="", bbox=bounds)
    return im1, im2


def read_unions(fname1, fname2, band1=None, band2=None):
    """Read in the union of 2 files as an array"""
    bounds = get_union_bounds(fname1, fname2)
    print(f"bounds: {bounds}")
    im1 = copy_vrt(fname1, out_fname="", bbox=bounds)
    im2 = copy_vrt(fname2, out_fname="", bbox=bounds)
    return im1, im2


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


def merge_files(file1, file2, deramp1=False, deramp2=False, deramp_order=2):
    """Create a merged version of two files, bounded by their union bounds"""

    img1, img2 = read_unions(file1, file2)
    if deramp1:
        mask1 = img1 == 0
        img1 = np.nan_to_num(remove_ramp(img1, deramp_order=deramp_order, mask=mask1))
    if deramp2:
        mask2 = img2 == 0
        img2 = np.nan_to_num(remove_ramp(img2, deramp_order=deramp_order, mask=mask2))
    valid1 = (img1 != 0).astype(int)
    valid2 = (img2 != 0).astype(int)
    valid_count = valid1 + valid2
    return (img1 + img2) / valid_count


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
    mkdir_p(output_dir)
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
        mkdir_p(os.path.join(output_dir, dirname))
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
                mkdir_p(newdir)
            else:
                newdir = os.path.join(output_dir, dirname)
            outname = os.path.join(newdir, filename)

            if os.path.exists(outname) and not overwrite:
                if verbose:
                    tqdm.write("%s exists, skipping" % outname)
                continue

            if verbose:
                tqdm.write("Subsetting %s to %s" % (f, outname))

            ulx, uly, lrx, lry = _get_bounds(bbox_rdr, fname=f, transform=transform)
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


def _get_bounds(bbox_rdr, fname=None, transform=None):
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
    # "unw_stack.h5",
    # "cor_stack.h5",
    "ifg_stack.h5",
    # "masks.h5",
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

    mkdir_p(output_dir)
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
    compressions = {ds: {"zlib": True} for ds in dsets_to_compress}
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
    project_dir=".",
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
    from apertools import sario,utils

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
        if not min_date.tzinfo:
            min_date = min_date.replace(tzinfo=datetime.timezone.utc)
    else:
        min_date = datetime.datetime(1900, 1, 1, tzinfo=datetime.timezone.utc)
    if max_date:
        if not max_date.tzinfo:
            max_date = max_date.replace(tzinfo=datetime.timezone.utc)
    else:
        max_date = datetime.datetime(2100, 1, 1, tzinfo=datetime.timezone.utc)

    ifglist = sario.load_ifglist_from_h5(files_to_crop[0])
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

    # slclist = sario.load_slclist_from_h5(files_to_crop[0])
    # slclist2019 = [slc for slc in slclist if min_date <= slc <= max_date]
    # idxs_slc = [slclist.index(ifg) for ifg in slclist2019]

    for fname in files_to_crop:
        logger.info("Subsetting %s", fname)
        filepath = os.path.join(project_dir, fname)
        ext = os.path.splitext(filepath)[1]
        f_out = filepath.replace(ext, out_str + ext)
        if os.path.exists(f_out):
            logger.info("%s exists: skipping", f_out)
            continue

        ds = xr.open_dataset(filepath, engine="h5netcdf", phony_dims="sort")
        # ds_sub = ds.sel(ifg_idx=ifg_idxs, phony_dim_4=idxs_slc)
        ds_sub = ds.sel(ifg_idx=ifg_idxs)
        logger.info("Input dimensions: %s", ds.dims)
        logger.info("Output dimensions: %s", ds_sub.dims)

        ###
        ###
        dsets_to_compress = COMP_DSETS[fname]
        compressions = {ds: {"zlib": True} for ds in dsets_to_compress}
        logger.info("Writing to %s, compressing with: %s", f_out, compressions)
        ds_sub.to_netcdf(f_out, engine="h5netcdf", encoding=compressions, mode="a")
