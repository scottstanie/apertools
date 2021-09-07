#!/usr/bin/env python3
import argparse
import os
import subprocess
import apertools.log

logger = apertools.log.get_log()


def geocode(
    infile=None,
    outfile=None,
    lat=None,
    lon=None,
    rows=None,
    cols=None,
    bbox=None,
    lon_step=0.0005,
    lat_step=0.0005,
    resampling="nearest",
    dtype="float32",
    cleanup=False,
):
    import rasterio.errors

    if not infile:
        raise ValueError("infile is required")
    is_hdf5_dset = infile.startswith("HDF5")
    if lat is None or lon is None:
        if not is_hdf5_dset:
            raise ValueError("lat, lon are required if `infile` is not HDF5")
        else:
            dirname, full_path, subdset = _abs_path_hdf5_string(infile)
            lat = full_path.replace(subdset, "//lat")
            lon = full_path.replace(subdset, "//lon")
            logger.info("Using lat/lon files: %s, %s", lat, lon)

    logger.info("Creating temp lat/lon VRT files.")
    tmp_lat_file, tmp_lon_file = prepare_lat_lon(lat, lon)
    if not bbox:
        logger.info("Finding bbox from lat/lon file")
        bbox = _get_bbox_from_files(tmp_lat_file, tmp_lon_file)
    bbox_str = "%f %f %f %f" % tuple(bbox)
    logger.info(
        f"Geocoding in {bbox = } with (lon, lat) step = ({lon_step, lat_step })"
    )

    try:
        rows, cols = _get_size(infile)
        dtype = _get_dtype(infile)
    except rasterio.errors.RasterioIOError:
        rows, cols, dtype = rows, cols, dtype

    if not rows or not cols or not dtype:
        raise ValueError(
            f"Could not get image size for {infile} from GDAL;"
            "must pass --rows, --cols, --dtype"
        )

    dirname, infile, _ = _abs_path_hdf5_string(infile)
    tmp_vrt_in_file = os.path.join(dirname, "temp_geoloc_data.vrt")
    logger.info("Saving VRT for %s to %s", infile, tmp_vrt_in_file)
    # writeVRT(infile, tmp_lat_file, tmp_lon_file)
    tmp_vrt_in_file = write_vrt(
        infile,
        tmp_vrt_in_file,
        tmp_lat_file,
        tmp_lon_file,
        rows,
        cols,
        dtype,
    )

    if not outfile:
        filename_only = os.path.split(infile.replace("HDF5:", "").split(":")[0])[0]
        outfile = os.path.join(dirname, filename_only) + ".geo"
    driver = "GTiff"
    if outfile.endswith(".tif"):
        driver = "GTiff"
    elif outfile.endswith(".vrt"):
        driver = "VRT"
    else:
        driver = "ENVI"
    logger.info("Geocoding output to %s to %s with driver %s", infile, outfile, driver)

    cmd = (
        # use the geolocation array to geocode, set target extent w/ bbox
        f"gdalwarp -geoloc -te {bbox_str} "
        # use specified lat/lon step for target resolution
        f" -tr {lon_step} {lat_step}"
        # Specify resampling method
        f" -r {resampling}"
        # Add warp option: run on multiple threads. Give the output a latlon projection
        ' -multi -wo GDAL_NUM_THREADS=ALL_CPUS -t_srs "+proj=longlat +datum=WGS84 +nodefs"'
        # Add ENVI header suffix, not replace (.unw.geo.hdr, not .unw.hdr)
        f" -of {driver} -co SUFFIX=ADD"
        # set nodata values on source and destination
        " -srcnodata 0 -dstnodata 0 "
        f" {tmp_vrt_in_file} {outfile}"
    )
    _log_and_run(cmd)

    if cleanup:
        for f in [tmp_lat_file, tmp_lon_file, tmp_vrt_in_file]:
            logger.info("Removing %s", f)
            os.remove(f)


def _read_4_corners(f, band=1):
    from rasterio.windows import Window
    import rasterio as rio

    rows, cols = _get_size(f)

    pixels = []
    with rio.open(f) as src:
        for offset in [(0, 0), (cols - 1, 0), (0, rows - 1), (cols - 1, rows - 1)]:
            pixel = src.read(band, window=Window(*offset, 1, 1))
            pixels.append(float(pixel))
    return pixels


def _get_dtype(f):
    import rasterio as rio

    with rio.open(f) as src:
        return src.meta["dtype"]


def _get_bbox_from_files(lat_file, lon_file):
    lon_corners = _read_4_corners(lon_file)
    left, right = min(lon_corners), max(lon_corners)

    lat_corners = _read_4_corners(lat_file)
    bot, top = min(lat_corners), max(lat_corners)

    return left, bot, right, top


def _get_size(f):
    import rasterio as rio

    with rio.open(f) as src:
        return src.shape


def _get_looks(f):
    """Get the row/col looks from the transform"""
    import rasterio as rio

    with rio.open(f) as src:
        x_step, _, _, _, y_step, _ = tuple(src.transform)[:6]
        # x_step is column looks, y_step is row looks
        return y_step, x_step


def _abs_path_hdf5_string(fname):
    """Get the absolute path and directory from (possibly) HDF5 gdal string
    If it's just a path name, returns the absolute path

    Args:
        fname (str): file path, or HDF5 path string

    Returns:
        dirname, full_path (loadable by gdal), sub-dataset

    Example:
        >>> _abs_path_hdf5_string('HDF5:file.h5://dataset_name')
        "/path/to/", "HDF5:/path/to/file.h5://dataset_name", "datset_name"

    """
    if "HDF5" in fname:
        driver, f, subdset = fname.split(":")
        full_path = os.path.abspath(f)
        dirname = os.path.dirname(full_path)
        return dirname, "{}:{}:{}".format(driver, full_path, subdset), subdset
    else:
        full_path = os.path.abspath(fname)
        dirname = os.path.dirname(fname)
        return dirname, full_path, None


def prepare_lat_lon(lat_file, lon_file):
    dirname, lat_file, _ = _abs_path_hdf5_string(lat_file)
    dirname, lon_file, _ = _abs_path_hdf5_string(lon_file)
    row_looks, col_looks = _get_looks(lat_file)
    print(f"{row_looks = } {col_looks = }")

    # temp_lat = lat_file + ".temp_geoloc.vrt"
    # temp_lon = lon_file + ".temp_geoloc.vrt"
    temp_lat = os.path.join(dirname, "temp_geoloc_lat.vrt")
    temp_lon = os.path.join(dirname, "temp_geoloc_lon.vrt")

    target_res = f"-tr {col_looks} {row_looks}" if "HDF5" not in lat_file else ""
    cmd = f"gdal_translate -of VRT -a_nodata 0 {target_res} {lat_file} {temp_lat} "
    _log_and_run(cmd)
    cmd = f"gdal_translate -of VRT -a_nodata 0 {target_res} {lon_file} {temp_lon} "
    _log_and_run(cmd)

    return temp_lat, temp_lon


def _log_and_run(cmd):
    logger.info(cmd)
    subprocess.run(cmd, shell=True, check=True)


# <metadata domain="GEOLOCATION">
# <mdi key="Y_DATASET">/data7/jpl/sanAnd_23511/testint/tempLAT.vrt</mdi>
# <mdi key="X_DATASET">/data7/jpl/sanAnd_23511/testint/tempLON.vrt</mdi>


def write_vrt(
    filename, outfile, lat_file, lon_file, rows, cols, dtype, row_looks=1, col_looks=1
):
    import apertools.sario

    metadata_dict = {
        "Y_DATASET": lat_file,
        "X_DATASET": lon_file,
        "X_BAND": "1",
        "Y_BAND": "1",
        "PIXEL_OFFSET": "0",
        "LINE_OFFSET": "0",
        "LINE_STEP": str(row_looks),
        "PIXEL_STEP": str(col_looks),
    }
    if "HDF5" not in filename:
        outfile = apertools.sario.save_vrt(
            filename,
            outfile=outfile,
            rows=rows,
            cols=cols,
            dtype=dtype,
            metadata_domain="GEOLOCATION",
            metadata_dict=metadata_dict,
        )
    else:
        from osgeo import gdal

        # Virst translate for base VRT, then add the -geoloc array data
        cmd = f"gdal_translate -of VRT {filename} {outfile} "
        _log_and_run(cmd)
        ds = gdal.Open(outfile, gdal.GA_Update)
        ds.SetMetadata(metadata_dict, "GEOLOCATION")
        ds = None

    return outfile


RESAMPLE_CHOICES = [
    "near",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "average",
    "med",
]


def get_cli_args():
    """
    Create command line parser.
    """

    parser = argparse.ArgumentParser(
        description="Geocode an image in radar coordinates using lat/lon files"
    )
    parser.add_argument(
        "infile",
        help="Name of input file to be geocoded.",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        help="Name of input file to be geocoded (default = `infile`.geo).",
    )
    parser.add_argument(
        "--lat",
        help="latitude file in radar coordinate (if not contained within `infile`)",
    )
    parser.add_argument(
        "--lon",
        help="longitude file in radar coordinate (if not contained within `infile`)",
    )
    parser.add_argument(
        "--rows", type=int, help="Number of rows of input files (if not GDAL readable)"
    )
    parser.add_argument(
        "--cols", type=int, help="Number of cols of input files (if not GDAL readable)"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        metavar=("left", "bottom", "right", "top"),
        type=float,
        help="Bounding box of area of interest "
        " (e.g. --bbox -106.1 30.1 -103.1 33.1 ). "
        "If none given, will get max extent of lat/lon file",
    )
    parser.add_argument(
        "--lon-step",
        default=0.001,
        help="output pixel size (longitude) in degrees (default = %(default)s)",
    )
    parser.add_argument(
        "--lat-step",
        default=0.001,
        help="output pixel size (latitude) in degrees (default = %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--resampling",
        default=RESAMPLE_CHOICES[0],
        choices=RESAMPLE_CHOICES,
        help="Resampling method (choices = %(choices)s, default=%(default)s)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "complex64", "int16", "uint8"],
        help="(numpy-style) data type of binary array "
        "(choices = %(choices)s, default=%(default)s)",
    )
    parser.add_argument("--cleanup", help="Remove temp .vrt files after finished")
    return parser.parse_args()

    # <metadata domain="GEOLOCATION">
    # <mdi key="Y_DATASET">/data7/jpl/sanAnd_23511/testint/tempLAT.vrt</mdi>
    # <mdi key="X_DATASET">/data7/jpl/sanAnd_23511/testint/tempLON.vrt</mdi>


def main():
    args = get_cli_args()
    arg_dict = vars(args)
    logger.info("Input arguments:")
    logger.info(arg_dict)
    geocode(**arg_dict)


if __name__ == "__main__":
    main()
