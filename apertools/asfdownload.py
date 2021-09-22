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
import datetime
import os
import subprocess
from collections import Counter

# import apertools.geojson
# from osgeo import gdal

# python 2: urllib.urlencode
from urllib.parse import urlencode

# urlencode({"intersectsWith":"point(-119.543 37.925)"})
# Out[51]: 'intersectsWith=point%28-119.543+37.925%29'
#
# Note: quote_via=urllib.parse.quote for quote spaces with percent

DIRNAME = os.path.dirname(os.path.abspath(__file__))


def form_url(
    bbox=None,
    dem=None,
    # geojson=None,
    wkt_file=None,
    start=None,
    end=None,
    processingLevel="SLC",
    relativeOrbit=None,
    absoluteOrbit=None,
    flightLine=None,
    maxResults=2000,
    query_filetype="geojson",
    platform="S1",
    beamMode="IW",
    **kwargs,
):
    """
    dem (str): Name of DEM filename (will parse bbox)
    bbox(tuple): lower left lon, lat, upper right format
        e.g. bbox=(-150.2,65.0,-150.1,65.5)
    processingLevel (str): options, "RAW", "SLC" for sentinel
    start (str): Starting time for search. Many acceptable inputs
        e.g. "3 months and a day ago" "May 30, 2018", "2010-10-30T11:59:59UTC"
    end (str): Ending time for search, see "start" for options
    query_filetype (str): default="geojson". options: "csv", "kml", "geojson"
    """
    if dem is not None:
        bbox = get_dem_bbox(dem)
    elif wkt_file is not None:
        bbox = get_wkt_bbox(wkt_file)

    # if bbox is None and absoluteOrbit is None:
    # raise ValueError("Need either bbox or dem options without absoluteOrbit")

    # TODO: geojson to WKT for intersection
    params = dict(
        bbox=",".join(map(str, bbox)) if bbox else None,
        start=start,
        end=end,
        processingLevel=processingLevel,
        relativeOrbit=relativeOrbit,
        absoluteOrbit=absoluteOrbit,
        flightLine=flightLine,
        maxResults=maxResults,
        output=query_filetype.upper(),
        platform=platform,
        beamMode=beamMode,
    )
    params = {k: v for k, v in params.items() if v is not None}
    base_url = "https://api.daac.asf.alaska.edu/services/search/param?{params}"
    return base_url.format(params=urlencode(params))


def query_only(query_filetype="geojson", **kwargs):
    # Save files into correct output type:
    outname = "asfquery.{}".format(query_filetype.lower())

    url = form_url(query_filetype=query_filetype, **kwargs)
    data_cmd = """ curl "{url}" > {outname} """.format(url=url, outname=outname)
    print("Running command:")
    print(data_cmd)
    subprocess.check_call(data_cmd, shell=True)
    return outname


def download_data(query_filetype="metalink", out_dir=".", **kwargs):
    # Start by saving data available as a metalink file
    outname = query_only(query_filetype=query_filetype, **kwargs)

    aria2_conf = os.path.expanduser("~/.aria2/asf.conf")
    download_cmd = (
        f"aria2c --http-auth-challenge=true --continue=true "
        f"--conf-path={aria2_conf} --dir={out_dir} {outname}"
    )
    print("Running command:")
    print(download_cmd)
    subprocess.check_call(download_cmd, shell=True)


def get_dem_bbox(fname):
    import rasterio as rio

    with rio.open(fname) as ds:
        # left, bottom, right, top = ds.bounds
        return ds.bounds


def get_wkt_bbox(fname):
    from shapely import wkt

    with open(fname) as f:
        return wkt.load(f).bounds


def parse_query_results(fname="asfquery.geojson"):
    """Extract the path number counts and date ranges from a geojson query result"""
    import geojson

    with open(fname) as f:
        results = geojson.load(f)
    features = results["features"]
    # In[128]: pprint(results["features"])
    # [{'geometry': {'coordinates': [[[-101.8248, 34.1248],...
    #            'type': 'Polygon'},
    # 'properties': {'beamModeType': 'IW', 'pathNumber': '85',
    # 'type': 'Feature'}, ...
    print(f"Found {len(features)} results:")
    if len(features) == 0:
        return Counter(), []

    # Include both the number and direction (asc/desc) in Counter key
    path_nums = Counter(
        [
            (f["properties"]["pathNumber"], f["properties"]["flightDirection"].lower())
            for f in features
        ]
    )
    print(f"Count by pathNumber: {path_nums.most_common()}")
    starts = Counter([f["properties"]["startTime"] for f in features])
    starts = [datetime.datetime.fromisoformat(s) for s in starts]
    print(f"Dates ranging from {min(starts)} to {max(starts)}")

    return path_nums, starts


def _platform_choices():
    import pandas as pd

    fname = os.path.join(DIRNAME, "data/asfquery_platforms.csv")
    return pd.read_csv(fname)


def _platform_beammodes():
    import pandas as pd

    fname = os.path.join(DIRNAME, "data/asfquery_platform_beammodes.csv")
    df = pd.read_csv(fname)
    df["values"] = df["values"].str.split(", ")
    return df


def _check_platform(args):
    choices = _platform_choices()
    if not (args.platform in choices["name"].values or args.platform is None):
        raise ValueError(f"Invalid platform: {args.platform}. Choices: {choices}")


def _check_beammode(args):
    platdf = _platform_choices()
    canonical_name_df = platdf[platdf["name"].str.lower() == args.platform.lower()]
    canonical_name = canonical_name_df.iloc[0]["canonical"]

    beamdf = _platform_beammodes()
    choices = beamdf[beamdf["platform"] == canonical_name]["values"].iloc[0]
    if not (args.beamMode in choices or args.beamMode is None):
        raise ValueError(
            f"Invalid beamMode {args.beamMode} for platform {args.platform}. "
            f"Choices: {choices}"
        )


def cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        "-o",
        help="Path to directory for saving output files (default=%(default)s)",
        default="./",
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        metavar=("left", "bottom", "right", "top"),
        type=float,
        help="Bounding box of area of interest "
        " (e.g. --bbox -106.1 30.1 -103.1 33.1 ). ",
    )
    p.add_argument(
        "--dem",
        help="Filename of a (gdal-readable) DEM",
    )
    p.add_argument(
        "--wkt-file",
        help="Filename of a WKT polygon to search within",
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
        "--platform",
        help="Remote sensing platform (default=%(default)s)",
        default="S1",
    )
    p.add_argument(
        "--processingLevel",
        default="SLC",
        help="Level or product to download (default=%(default)s)",
    )
    p.add_argument(
        "--beamMode",
        help="Type of acquisition mode for data (default=%(default)s)",
        default="IW",
    )
    p.add_argument(
        "--relativeOrbit",
        type=int,
        help="Limit to one path / relativeOrbit",
    )
    p.add_argument(
        "--flightLine",
        type=int,
        help="UAVSAR flight line",
    )
    p.add_argument(
        "--absoluteOrbit",
        type=int,
        help="Either orbit cycle count, or (for UAVSAR) the flightId",
    )
    p.add_argument(
        "--maxResults",
        type=int,
        default=2000,
        help="Limit of number of products to download (default=%(default)s)",
    )
    p.add_argument(
        "--query-only",
        action="store_true",
        help="display available data in format of --query-file, no download",
    )
    p.add_argument(
        "--query-file",
        default="metalink",
        choices=["metalink", "geojson", "kml"],
        help="Type of output file to save query to (default=%(default)s)",
    )
    args = p.parse_args()
    if all(vars(args)[item] for item in ("bbox", "dem", "absoluteOrbit", "flightLine")):
        raise ValueError(
            "Need either --bbox or --dem options without flightLine/absoluteOrbit"
        )

    _check_platform(args)
    _check_beammode(args)

    if args.query_only:
        query_only(**vars(args))
    else:
        download_data(**vars(args))


if __name__ == "__main__":
    cli()
