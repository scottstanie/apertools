#!/usr/bin/env python3
import argparse
import os
from osgeo import gdal
import subprocess

# import xml.etree.ElementTree as ET
import apertools.log

logger = apertools.log.get_log()


def geocode(args):

    lat_file, lon_file = prepare_lat_lon(args)
    if not args.bbox:
        bbox = _get_bbox(lat_file, lon_file)
    else:
        bbox = args.bbox

    for infile in args.file_list:
        infile = os.path.abspath(infile)
        print("geocoding " + infile)
        writeVRT(infile, lat_file, lon_file)
        write_vrt(infile, lat_file, lon_file, args.rows, args.cols, args.dtype)

        outfile = infile + ".geo"
        cmd = (
            f"gdalwarp -of ENVI -geoloc -te {bbox} "
            f" -tr {args.lon_step} {args.lat_step}"
            " -srcnodata 0 -dstnodata 0 "
            f" -r {args.resamplingMethod}"
            f" {infile} {outfile}"
        )
        _log_and_run(cmd)


#
def _get_bbox(lat_file, lon_file):
    ds = gdal.Open(lat_file)
    # min, max, mean, stddev
    bot, top, _, _ = ds.GetRasterBand(1).GetStatistics(0, 1)
    ds = None
    ds = gdal.Open(lon_file)
    left, right, _, _ = ds.GetRasterBand(1).GetStatistics(0, 1)
    ds = None
    return left, bot, right, top


def prepare_lat_lon(args):
    lat_file = os.path.abspath(args.lat)
    lon_file = os.path.abspath(args.lon)
    # Need to save a vrt? or just translate?
    # cmd = f"aper save-vrt {lat_file}"
    # _log_and_run(cmd)
    # cmd = f"aper save-vrt {lon_file}"
    # _log_and_run(cmd)

    tempLat = os.path.join(os.path.dirname(args.file_list[0]), "tempLAT.vrt")
    tempLon = os.path.join(os.path.dirname(args.file_list[0]), "tempLON.vrt")

    cmd = f"gdal_translate -of VRT {lat_file} {tempLat} -a_nodata 0 "
    _log_and_run(cmd)
    cmd = f"gdal_translate -of VRT {lon_file} {tempLon} -a_nodata 0 "
    _log_and_run(cmd)

    return tempLat, tempLon


def _log_and_run(cmd):
    logger.info(cmd)
    subprocess.run(cmd, shell=True, check=True)


def writeVRT(infile, lat_file, lon_file):
    # This function is modified from isce2gis.py
    lat_file = os.path.abspath(lat_file)
    lon_file = os.path.abspath(lon_file)
    infile = os.path.abspath(infile)
    cmd = "isce2gis.py vrt -i " + infile
    os.system(cmd)

    tree = ET.parse(infile + ".vrt")
    root = tree.getroot()

    meta = ET.SubElement(root, "metadata")
    meta.attrib["domain"] = "GEOLOCATION"
    meta.tail = "\n"
    meta.text = "\n    "

    rdict = {
        "Y_DATASET": lat_file,
        "X_DATASET": lon_file,
        "X_BAND": "1",
        "Y_BAND": "1",
        "PIXEL_OFFSET": "0",
        "LINE_OFFSET": "0",
        "LINE_STEP": "1",
        "PIXEL_STEP": "1",
    }

    for key, val in rdict.items():
        data = ET.SubElement(meta, "mdi")
        data.text = val
        data.attrib["key"] = key
        data.tail = "\n    "

    data.tail = "\n"
    tree.write(infile + ".vrt")


def write_vrt(filename, lat_file, lon_file, rows, cols, dtype, row_looks=1, col_looks=1):
    import apertools.sario

    apertools.sario.save_vrt(
        filename,
        rows=rows,
        cols=cols,
        dtype=dtype,
        num_bands=1,
        metadata_domain="GEOLOCATION",
        metadata_dict={
            "Y_DATASET": lat_file,
            "X_DATASET": lon_file,
            "X_BAND": "1",
            "Y_BAND": "1",
            # "PIXEL_OFFSET": "0",
            # "LINE_OFFSET": "0",
            "LINE_STEP": str(row_looks),
            "PIXEL_STEP": str(col_looks),
        },
    )

    # <metadata domain="GEOLOCATION">
    # <mdi key="Y_DATASET">/data7/jpl/sanAnd_23511/testint/tempLAT.vrt</mdi>
    # <mdi key="X_DATASET">/data7/jpl/sanAnd_23511/testint/tempLON.vrt</mdi>


def get_size(f):
    ds = gdal.Open(f, gdal.GA_ReadOnly)
    b = ds.GetRasterBand(1)
    width = b.XSize
    length = b.YSize
    ds = None
    return width, length


def get_bounds(filename):
    import rasterio as rio

    with rio.open(filename) as src:
        return src.bounds


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
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
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
    return parser.parse_args()

    # <metadata domain="GEOLOCATION">
    # <mdi key="Y_DATASET">/data7/jpl/sanAnd_23511/testint/tempLAT.vrt</mdi>
    # <mdi key="X_DATASET">/data7/jpl/sanAnd_23511/testint/tempLON.vrt</mdi>


def main():
    args = get_cli_args()
    geocode(args)


if __name__ == "__main__":
    main()
