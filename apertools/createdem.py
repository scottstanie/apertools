#!/usr/bin/env python
"""
Command line interface for running createdem
"""
import sys
import glob
import os
import json
import argparse
from argparse import (
    ArgumentError,
    ArgumentParser,
    ArgumentTypeError,
    FileType,
    RawTextHelpFormatter,
)
import subprocess

# from osgeo import gdal

PATH = os.path.dirname(os.path.abspath(sys.argv[0]))


def positive_small(argstring):
    try:
        # val = int(argstring)
        val = float(argstring)
        assert val > 0 and val < 50
    except (ValueError, AssertionError):
        raise ArgumentTypeError("--rate must be positive integer < 50")
    return val


DESCRIPTION = """Form a cropped (upsampled) DEM from SRTM GL1

    Uses SRTM from sdsc.edu opentopography

    GDAL downloads and crops/upsamples/translates to ENVI format


    Usage Examples:

        createdem -156.0 20.2 1 2 --rate 2  # Makes a box 1 degree wide, 2 deg high
        createdem -156.0 20.2 -154.5 21.4 -o my_elevation.dem
        createdem --geojson ../aoi.geojson --xrate 2 --yrate 2
        createdem --wkt "POLYGON((-104.8 31.2,-103.4 31.2,-103.4 32.2,-104.8 32.2,-104.8 31.2))"

"""


def cli():
    parser = ArgumentParser(
        prog="createdem",
        description=DESCRIPTION,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        metavar=("left", "bottom", "right", "top"),
        type=float,
        help="Bounding box of area of interest "
        " (e.g. --bbox -106.1 30.1 -103.1 33.1 ). ",
    )
    parser.add_argument(
        "--geojson",
        "-g",
        type=FileType(),
        help="Alternate to bbox specification: \n"
        "File containing the geojson object for DEM bounds",
    )
    parser.add_argument(
        "--wkt",
        help="Alternate to bbox specification: \n"
        "String of well known text (WKT) for DEM area",
    )
    parser.add_argument(
        "--xrate",
        "-x",
        default=1,
        type=positive_small,
        help="Upsample DEM in x (range) direction (default=%(default)s)",
    )
    parser.add_argument(
        "--yrate",
        "-y",
        default=1,
        type=positive_small,
        help="Upsample DEM in x (range) direction (default=%(default)s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="elevation.dem",
        help="Name of output dem file (default=%(default)s)",
    )

    args = parser.parse_args()
    if args.bbox and args.geojson:
        raise ArgumentError(args.geojson, "Can only specify one area type")
    elif not args.bbox and not args.geojson and not args.wkt:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    import rasterio.features
    import shapely.wkt

    if args.bbox is not None:
        left, bottom, right, top = args.bbox
    elif args.geojson is not None:
        left, bottom, right, top = rasterio.features.bounds(json.load(args.geojson))
    elif args.wkt is not None:
        left, bottom, right, top = rasterio.features.bounds(shapely.wkt.loads(args.wkt))

    main(left, bottom, right, top, args.xrate, args.yrate, args.output)


def main(left, bottom, right, top, xrate=1, yrate=1, outname="elevation.dem"):
    import rasterio

    print("Boundaries, xrate, yrate")
    print(left, bottom, right, top, xrate, yrate)
    xres = 1 / 3600 / xrate
    yres = 1 / 3600 / yrate

    # For gdal, the windows are by pixel edges, not centers
    # If we want it like one SRTM tile (size 3601x3601), need to expand rect by half a box
    # left -= abs(xres) / 2
    # right += abs(xres) / 2
    # top += abs(yres) / 2
    # bottom -= abs(yres) / 2
    # print((bottom, top, left, right))

    # Without this, we get a box (3600x3600) for a 1 deg window
    # which turns out cleaner for taking looks later (no cutting off last pixel)

    # add small padding for new API. use gdal_translate later for exact bounds
    # API can only give 30m pixel bounds (instead of upsampled window bounds)
    pleft = left - 0.1
    pbottom = bottom - 0.1
    pright = right + 0.1
    ptop = top + 0.1

    srtm_url = (
        f"https://portal.opentopography.org/API/globaldem?"
        f"demtype=SRTMGL1_E&west={pleft}&south={pbottom}&east={pright}&north={ptop}"
        "&outputFormat=GTiff"
    )
    command = f'curl -o tmp_elevation.tif "{srtm_url}"'
    print(command)
    subprocess.check_call(command, shell=True)

    # https://gdal.org/programs/gdal_translate.html
    # -of <format> Select the output format.
    # -ot <type> Force the output image bands to have a specific type. Use type names
    # -tr <xres> <yres> set target resolution in georeferenced units
    # -r resampling method
    # -projwin <ulx> <uly> <lrx> <lry> Selects a subwindow from the source image for copying
    command = (
        f"gdal_translate -of ENVI -ot Int16 -tr {xres:.15f} {yres:.15f} -a_nodata -32768 "
        f"-projwin {left} {top} {right} {bottom} "
        "-r bilinear tmp_elevation.tif tmp_elevation.dem"
    )

    command = command.format(
        xres=xres,
        yres=yres,
    )
    print(command)
    subprocess.check_call(command, shell=True)

    # Set nodata (-32768) to 0
    command = (
        f'gdal_calc.py --quiet --NoDataValue=0 --calc="A*(A!=-32768)" '
        f"-A tmp_elevation.dem --outfile={outname} --format=ENVI"
    )
    print(command)
    subprocess.check_call(command, shell=True)

    for f in glob.glob("tmp_elevation*"):
        os.remove(f)

    # ds = gdal.Open('elevation.dem')
    # trans = ds.GetGeoTransform()
    # print(trans)
    # width = ds.RasterXSize
    # length = ds.RasterYSize
    # See here for geotransform info
    # https://gdal.org/user/raster_data_model.html#affine-geotransform
    # The affine transform consists of six coefficients returned by
    # GDALDataset::GetGeoTransform() which map pixel/line coordinates
    # into georeferenced space using the following relationship:
    # Set the .rsc file to start in the center of the first pixel (instead of edge)
    # X0 = trans[0] + .5 * trans[1] + .5 * trans[2]
    # Y0 = trans[3] + .5 * trans[4] + .5 * trans[5]
    # print((X0, Y0, xsize, ysize))

    # rasterio way:
    ds = rasterio.open(outname)
    length, width = ds.shape
    # affine.Affine(a, b, c,
    #               d, e, f)
    # e.g.: Affine(0.000277777777, 0.0, -104.10000000252,
    #              0.0, -0.000277777777, 32.80000000056)
    # GDAL geotransform looks like:
    # (c, a, b, f, d, e)
    x_step, _, x_edge, _, y_step, y_edge, _, _, _ = tuple(ds.transform)
    ds.close()
    X0 = x_edge + 0.5 * x_step
    Y0 = y_edge + 0.5 * y_step
    print(X0, Y0, width, length)

    #  make a rsc file for processing
    fd = open(outname + ".rsc", "w")
    fd.write("WIDTH         " + str(width) + "\n")
    fd.write("FILE_LENGTH   " + str(length) + "\n")
    fd.write("X_FIRST       " + str(X0) + "\n")
    fd.write("Y_FIRST       " + str(Y0) + "\n")
    fd.write("X_STEP        " + str(x_step) + "\n")
    fd.write("Y_STEP        " + str(y_step) + "\n")
    fd.write("X_UNIT        degrees\n")
    fd.write("Y_UNIT        degrees\n")
    fd.write("Z_OFFSET      0\n")
    fd.write("Z_SCALE       1\n")
    fd.write("PROJECTION    LL\n")

    fd.close()


if __name__ == "__main__":
    cli()
