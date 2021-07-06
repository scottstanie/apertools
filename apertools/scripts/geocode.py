#!/usr/bin/env python3
import argparse
import os
import subprocess
import apertools.log

logger = apertools.log.get_log()


def geocode(args):

    logger.info("Createing lat/lon VRT files.")
    lat_file, lon_file = prepare_lat_lon(args, row_looks=args.row_looks, col_looks=args.col_looks)
    if not args.bbox:
        logger.info("Finding bbox from lat/lon file")
        bbox = _get_bbox_from_files(lat_file, lon_file)
    else:
        bbox = args.bbox
    bbox_str = '%f %f %f %f' % tuple(bbox)
    logger.info(f"Geocoding in {bbox = } with (lon, lat) step = ({args.lon_step, args.lat_step })")

    for infile in args.file_list:
        try:
            rows, cols = _get_size(infile)
            dtype = _get_dtype(infile)
        except RuntimeError:
            rows, cols, dtype = args.rows, args.cols, args.dtype
        if not rows or not cols or dtype:
            raise ValueError("Could not get image size for {infile} from GDAL;"
                             "must pass --rows, --cols, --dtype")

        infile = os.path.abspath(infile)
        print("geocoding " + infile)
        # writeVRT(infile, lat_file, lon_file)
        vrt_in_file = write_vrt(
            infile,
            lat_file,
            lon_file,
            args.rows,
            args.cols,
            args.dtype,
        )

        outfile = infile + ".geo"
        cmd = (
            f"gdalwarp -of ENVI -geoloc -te {bbox_str} "
            f" -tr {args.lon_step} {args.lat_step}"
            " -srcnodata 0 -dstnodata 0 "
            f" -r {args.resampling}"
            f" {vrt_in_file} {outfile}"
        )
        _log_and_run(cmd)


#
def _get_bbox_from_files(lat_file, lon_file):
    # TODO: pick out cornrs...
    rasterio.rio.insp.stats((src, 1))
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(lat_file)
    # min, max, mean, stddev
    bot, top, _, _ = ds.GetRasterBand(1).GetStatistics(0, 1)
    ds = None
    ds = gdal.Open(lon_file)
    left, right, _, _ = ds.GetRasterBand(1).GetStatistics(0, 1)
    ds = None
    return left, bot, right, top

def _get_size(f):
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(f)
    band = ds.GetRasterBand(1)
    rows = band.YSize
    cols = band.XSize
    ds = None
    return rows, cols

def prepare_lat_lon(args, row_looks=1, col_looks=1):
    lat_file = os.path.abspath(args.lat)
    lon_file = os.path.abspath(args.lon)

    temp_lat = os.path.join(lat_file, ".geo_lat.vrt")
    temp_lon = os.path.join(lon_file, ".geo_lon.vrt")

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
    filename, lat_file, lon_file, rows, cols, dtype, row_looks=1, col_looks=1
):
    import apertools.sario

    outfile = apertools.sario.save_vrt(
        filename,
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


def get_cli_args():
    """
    Create command line parser.
    """

    parser = argparse.ArgumentParser(
        description="Geocode an image in radar coordinates using lat/lon files"
    )
    parser.add_argument(
        "file_list",
        nargs="*",
        help="List of input files to be geocoded.",
    )
    parser.add_argument("--rows", type=int, help="Number of rows of input files (if not GDAL readable)")
    parser.add_argument("--cols", type=int, help="Number of cols of input files (if not GDAL readable)")
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
        default="bilinear",
        choices=["bilinear", "nearest", "cubic"],
        help="Resampling method (choices = %(choices)s, default=%(default)s)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "complex32", "int16", "uint8"],
        help="(numpy-style) data type of binary array "
        "(choices = %(choices)s, default=%(default)s)",
    )
    return parser.parse_args()

    # <metadata domain="GEOLOCATION">
    # <mdi key="Y_DATASET">/data7/jpl/sanAnd_23511/testint/tempLAT.vrt</mdi>
    # <mdi key="X_DATASET">/data7/jpl/sanAnd_23511/testint/tempLON.vrt</mdi>


def main():
    args = get_cli_args()
    geocode(args)


if __name__ == "__main__":
    main()
