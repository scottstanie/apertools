#!/usr/bin/env python
"""
Script for downloading through https://asf.alaska.edu/api/

Base taken from
https://github.com/scottyhq/isce_notes/blob/master/BatchProcessing.md


To download, you need aria2
yum install -y aria2

and either a .netrc:

# cat ~/.netrc
machine urs.earthdata.nasa.gov
    login CHANGE
    password CHANGE

or, an aria2 conf file

# $HOME/.aria2/asf.conf
http-user=CHANGE
http-passwd=CHANGE

max-concurrent-downloads=5
check-certificate=false

allow-overwrite=false
auto-file-renaming=false
always-resume=true
"""
import argparse
import os
import subprocess
# import apertools.geojson
import gdal

# python 2: urllib.urlencode
from urllib.parse import urlencode
# urlencode({"intersectsWith":"point(-119.543 37.925)"})
# Out[51]: 'intersectsWith=point%28-119.543+37.925%29'
#
# Note: quote_via=urllib.parse.quote for quote spaces with percent


def form_url(
    bbox=None,
    # geojson=None,
    start=None,
    end=None,
    processingLevel="RAW",
    relativeOrbit=None,
    maxResults=2000,
    output="geojson",
    platform="S1",
    beamMode="IW",
    **kwargs,
):
    """
    bbox(tuple): lower left lon, lat, upper right format
        e.g. bbox=(-150.2,65.0,-150.1,65.5)
    processingLevel (string): options, "RAW", "SLC" for sentinel
    start (string): Starting time for search. Many acceptable inputs
        e.g. "3 months and a day ago" "May 30, 2018", "2010-10-30T11:59:59UTC"
    end (string): Ending time for search, see "start" for options
    output (string): default="geojson". options: "csv", "kml", "geojson"
    """
    # TODO: geojson to WKT for intersection
    params = dict(
        bbox=",".join(map(str, bbox)),
        start=start,
        end=end,
        processingLevel=processingLevel,
        relativeOrbit=relativeOrbit,
        maxResults=maxResults,
        output=output.upper(),
        platform=platform,
        beamMode=beamMode,
    )
    params = {k: v for k, v in params.items() if v is not None}
    base_url = "https://api.daac.asf.alaska.edu/services/search/param?{params}"
    return base_url.format(params=urlencode(params))


def query_only(output="geojson", **kwargs):
    # Save files into correct output type:
    outname = "asfquery.{}".format(output.lower())

    url = form_url(output=output, **kwargs)
    data_cmd = """ curl "{url}" > {outname} """.format(url=url, outname=outname)
    print("Running command:")
    print(data_cmd)
    subprocess.check_call(data_cmd, shell=True)
    return outname


def downlaod_data(output="metalink", **kwargs):
    # Start by saving data available as a metalink file
    outname = query_only(output="metalink", **kwargs)

    aria2_conf = os.path.expanduser("~/.aria2/asf.conf")
    download_cmd = """aria2c --http-auth-challenge=true --conf-path={} {}""".format(
        aria2_conf,
        outname,
    )
    print("Running command:")
    print(download_cmd)
    subprocess.check_call(download_cmd, shell=True)


def get_ullr(filename):
    infodict = gdal.Info(filename, format="json")
    cc = infodict["cornerCoordinates"]
    ulx, uly = cc["upperLeft"]
    lrx, lry = cc["lowerRight"]
    return ulx, uly, lrx, lry


def ullr_to_bbox(ullr_tuple):
    """Convert gdal's ullr tuple to ASF bbox ( lower left lon, lat, upper right format

        Examples:
        >>> ullr=(-150.2, 65.5, -150.0, 65.0)
        >>> print(ullr_to_bbox(ullr))
        (-150.2, 65.0, -150.0, 65.5)
    """
    return ullr_tuple[0], ullr_tuple[3], ullr_tuple[2], ullr_tuple[1]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Only care for now about platform="S1",
    # Only care for now about beamMode="IW",
    p.add_argument(
        "--query-only",
        action="store_true",
        help="display available data in format of --output, no download",
    )
    p.add_argument(
        "--ullr",
        nargs=4,
        metavar=("ulx", "uly", "lrx", "lry"),
        type=float,
        help="Upper left, lower right coords "
        " (e.g. --ullr -106.1 33.1 -103.1, 30.1 ). "
        " Can be from gdalinfo on existing DEM.",
    )
    p.add_argument(
        "--dem",
        help="Filename of a (gdal-readable) DEM",
    )
    p.add_argument(
        "--start",
        help="Starting date for query (recommended: YYYY-MM-DD)",
    )
    p.add_argument(
        "--end",
        help="Ending date for query (recommended: YYYY-MM-DD)",
    )
    p.add_argument(
        "--output",
        "-o",
        default="kml",
        help="Type of output file to save query to (default=%(default)s)",
    )
    p.add_argument(
        "--processingLevel",
        choices=["RAW", "SLC"],
        default="RAW",
        help="Level or product to download (default=%(default)s)",
    )
    p.add_argument(
        "--relativeOrbit",
        type=int,
        help="Limit to one path / relativeOrbit",
    )
    p.add_argument(
        "--maxResults",
        type=int,
        default=2000,
        help="Limit of number of products to download (default=%(default)s)",
    )
    args = p.parse_args()
    if args.ullr is None and args.dem is None:
        raise ValueError("Need either --ullr or --dem options")

    arg_dict = vars(args)
    if args.dem:
        ullr = get_ullr(args.dem)
        arg_dict["ullr"] = ullr
    arg_dict["bbox"] = ullr_to_bbox(arg_dict["ullr"])

    if args.query_only:
        query_only(**vars(args))
    else:
        downlaod_data(**vars(args))
