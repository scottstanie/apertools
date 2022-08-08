#!/usr/bin/env python
import sys

import pandas as pd

from apertools import asfdownload, log


def main():
    logger = log.get_log()
    try:
        wkt_file = sys.argv[1]
    except IndexError:
        logger.error("No WKT file provided")
        raise
    try:
        out_dir = sys.argv[2]
    except IndexError:
        out_dir = "."
    logger.info(f"Downloading S1 metadata from WKT file {wkt_file} into {out_dir}")

    today = str(pd.Timestamp.today().date())
    biweekly_dates = pd.date_range(start="2014-09-01", end=today, freq="2W").strftime(
        "%Y-%m-%d"
    )

    for start, end in zip(biweekly_dates[:-1], biweekly_dates[1:]):
        logger.info(f"Downloading {start} to {end}")
        try:
            asfdownload.download_data(
                out_dir=".",
                wkt_file=wkt_file,
                start=start,
                end=end,
                processingLevel="METADATA_SLC",
                maxResults=2000,
                platform="S1",
                beamMode="IW",
            )
        except Exception:
            logger.error(f"Error downloading data from {start} to {end}", exc_info=True)
            continue


def get_exterior(xml_filename):
    from bs4 import BeautifulSoup
    from shapely import geometry
    soup = BeautifulSoup(open(xml_filename), "xml")

    coords = []
    for c in soup.find(name='gml:LinearRing').children:
        if not c.text.strip():
            continue
        coords.append(tuple(map(float, c.text.split())))
    return geometry.Polygon(coords)
    # poly.wkt


if __name__ == "__main__":
    main()
