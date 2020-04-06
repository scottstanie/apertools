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
import rasterio
import rasterio.features
import shapely.wkt

PATH = os.path.dirname(os.path.abspath(sys.argv[0]))


def positive_small_int(argstring):
    try:
        intval = int(argstring)
        assert (intval > 0 and intval < 50)
    except (ValueError, AssertionError):
        raise ArgumentTypeError("--rate must be positive integer < 50")
    return intval


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
        prog='createdem',
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
        type=positive_small_int,
        help="Upsample DEM in x (range) direction (default=%(default)s)",
    )
    parser.add_argument(
        "--yrate",
        "-y",
        default=1,
        type=positive_small_int,
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

    if args.bbox is not None:
        left, bottom, right, top = args.bbox
    elif args.geojson is not None:
        left, bottom, right, top = rasterio.features.bounds(json.load(args.geojson))
    elif args.wkt is not None:
        left, bottom, right, top = rasterio.features.bounds(shapely.wkt.loads(args.wkt))

    main(left, bottom, right, top, args.xrate, args.yrate, args.output)


def main(left, bottom, right, top, xrate=1, yrate=1, outname="elevation.dem"):
    print(left, bottom, right, top, xrate, yrate)
    xres = (1 / 3600 / xrate)
    yres = (1 / 3600 / yrate)

    # For gdal, the windows are by pixel edges, not centers, so expand rect by half a box
    left -= abs(xres) / 2
    right += abs(xres) / 2
    top += abs(yres) / 2
    bottom -= abs(yres) / 2
    print((bottom, top, left, right))

    srtm_url = "https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1_Ellip/SRTM_GL1_Ellip_srtm.vrt"  # noqa
    vsi_url = "/vsicurl/{}".format(srtm_url)

    # https://gdal.org/programs/gdal_translate.html
    # -of <format> Select the output format.
    # -ot <type> Force the output image bands to have a specific type. Use type names
    #  -tr <xres> <yres> set target resolution in georeferenced units
    # -r resampling method
    # -projwin <ulx> <uly> <lrx> <lry> Selects a subwindow from the source image for copying
    command = "gdal_translate -of ENVI -ot Int16 -tr {xres:.15f} {yres:.15f} -a_nodata -32768 \
-r bilinear -projwin {left} {top} {right} {bottom} {url} elevation_tmp.dem"

    command = command.format(
        xres=xres,
        yres=yres,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        url=vsi_url,
    )
    print(command)
    subprocess.check_call(command, shell=True)

    # Set nodata (-32768) to 0
    command = """gdal_calc.py --quiet --NoDataValue=0 --calc="A*(A!=-32768)" \
-A elevation_tmp.dem --outfile={} --format=ENVI""".format(outname)
    print(command)
    subprocess.check_call(command, shell=True)

    for f in glob.glob("elevation_tmp*"):
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
    ds = rasterio.open("elevation.dem")
    length, width = ds.shape
    # affine.Affine(a, b, c,
    #               d, e, f)
    # e.g.: Affine(0.000277777777, 0.0, -104.10000000252,
    #              0.0, -0.000277777777, 32.80000000056)
    # GDAL geotransform looks like:
    # (c, a, b, f, d, e)
    x_step, _, x_edge, _, y_step, y_edge, _, _, _ = tuple(ds.transform)
    X0 = x_edge + .5 * x_step
    Y0 = y_edge + .5 * y_step
    print(X0, Y0, width, length)

    #  make a rsc file for processing
    fd = open('elevation.dem.rsc', 'w')
    fd.write('WIDTH         ' + str(width) + "\n")
    fd.write('FILE_LENGTH   ' + str(length) + "\n")
    fd.write('X_FIRST       ' + str(X0) + "\n")
    fd.write('Y_FIRST       ' + str(Y0) + "\n")
    fd.write('X_STEP        ' + str(x_step) + "\n")
    fd.write('Y_STEP        ' + str(y_step) + "\n")
    fd.write('X_UNIT        degrees\n')
    fd.write('Y_UNIT        degrees\n')
    fd.write('Z_OFFSET      0\n')
    fd.write('Z_SCALE       1\n')
    fd.write('PROJECTION    LL\n')

    fd.close()


if __name__ == "__main__":
    cli()
