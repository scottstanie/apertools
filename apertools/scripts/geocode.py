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
    row_looks=1,
    col_looks=1,
    bbox=None,
    lon_step=0.0005,
    lat_step=0.0005,
    resampling="nearest",
    dtype="float32",
):
    import rasterio.errors

    if not infile or not lat or not lon:
        raise ValueError("infile, lat, lon are required")

    logger.info("Createing lat/lon VRT files.")
    lat_file, lon_file = prepare_lat_lon(
        lat, lon, row_looks=row_looks, col_looks=col_looks
    )
    if not bbox:
        logger.info("Finding bbox from lat/lon file")
        bbox = _get_bbox_from_files(lat_file, lon_file)
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

    infile = os.path.abspath(infile)
    vrt_in_file = infile + ".temp_geoloc.vrt"
    logger.info("Saving VRT for %s to %s", infile, vrt_in_file)
    # writeVRT(infile, lat_file, lon_file)
    vrt_in_file = write_vrt(
        infile,
        vrt_in_file,
        lat_file,
        lon_file,
        rows,
        cols,
        dtype,
    )

    if not outfile:
        outfile = infile + ".geo"
    logger.info("Geocoding output to %s to %s", infile, outfile)
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
        " -of ENVI -co SUFFIX=ADD"
        # set nodata values on source and destination
        " -srcnodata 0 -dstnodata 0 "
        f" {vrt_in_file} {outfile}"
    )
    _log_and_run(cmd)


def _read_4_corners(f, band=1):
    from rasterio.windows import Window
    import rasterio as rio

    rows, cols = _get_size(f)

    pixels = []
    with rio.open(f) as src:
        for offset in [(0, 0), (cols - 1, 0), (0, rows - 1), (cols - 1, rows - 1)]:
            w = Window(*offset, 1, 1)
            pixel = src.read(band, window=Window(*offset, 1, 1))
            pixels.append(float(pixel))
    return pixels


def _get_dtype(f):
    import rasterio as rio

    with rio.open(f) as src:
        return src.meta["dtype"]


def _get_bbox_from_files(lat_file, lon_file):
    # TODO: pick out cornrs...
    lon_corners = _read_4_corners(lon_file)
    left, right = min(lon_corners), max(lon_corners)

    lat_corners = _read_4_corners(lat_file)
    bot, top = min(lat_corners), max(lat_corners)

    return left, bot, right, top


def _get_size(f):
    import rasterio as rio

    with rio.open(f) as src:
        return src.shape


def prepare_lat_lon(lat_file, lon_file, row_looks=1, col_looks=1):
    lat_file = os.path.abspath(lat_file)
    lon_file = os.path.abspath(lon_file)

    temp_lat = lat_file + ".temp_geoloc.vrt"
    temp_lon = lon_file + ".temp_geoloc.vrt"

    cmd = f"gdal_translate -of VRT -a_nodata 0 -tr {col_looks} {row_looks} {lat_file} {temp_lat} "
    _log_and_run(cmd)
    cmd = f"gdal_translate -of VRT -a_nodata 0 -tr {col_looks} {row_looks} {lon_file} {temp_lon} "
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

    outfile = apertools.sario.save_vrt(
        filename,
        outfile=outfile,
        rows=rows,
        cols=cols,
        dtype=dtype,
        metadata_domain="GEOLOCATION",
        metadata_dict={
            "Y_DATASET": lat_file,
            "X_DATASET": lon_file,
            "X_BAND": "1",
            "Y_BAND": "1",
            "PIXEL_OFFSET": "0",
            "LINE_OFFSET": "0",
            "LINE_STEP": str(row_looks),
            "PIXEL_STEP": str(col_looks),
        },
    )
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
        "--rows", type=int, help="Number of rows of input files (if not GDAL readable)"
    )
    parser.add_argument(
        "--cols", type=int, help="Number of cols of input files (if not GDAL readable)"
    )
    parser.add_argument("--row-looks", default=1)
    parser.add_argument("--col-looks", default=1)
    parser.add_argument(
        "--lat",
        required=True,
        help="latitude file in radar coordinate",
    )
    parser.add_argument(
        "--lon",
        required=True,
        help="longitude file in radar coordinate",
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
