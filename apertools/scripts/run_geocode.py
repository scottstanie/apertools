#!/usr/bin/env python3
import argparse
import apertools.log
from apertools import geocode

logger = apertools.log.get_log()


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
        "--looks",
        type=int,
        nargs=2,
        help="Looks for input file, beyond with the --lat-file' and '--lon-file' show.",
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


def main():
    args = get_cli_args()
    print(args.looks)
    arg_dict = vars(args)
    logger.info("Input arguments:")
    logger.info(arg_dict)
    geocode.geocode(**arg_dict)


if __name__ == "__main__":
    main()
