#!/usr/bin/env python
from __future__ import annotations

import argparse

import pandas as pd
import backoff

from apertools import asfdownload, log


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def download_with_retry(
    out_dir: str,
    wkt_file: str,
    start: str,
    end: str,
    processingLevel: str,
    maxResults: int,
    platform: str,
    beamMode: str,
) -> None:
    asfdownload.download_data(
        out_dir=out_dir,
        wkt_file=wkt_file,
        start=start,
        end=end,
        processingLevel=processingLevel,
        maxResults=maxResults,
        platform=platform,
        beamMode=beamMode,
    )


def main() -> None:
    """Download Sentinel-1 metadata from a WKT file."""
    logger = log.get_log()

    parser = argparse.ArgumentParser(
        description="Download S1 metadata from a WKT file."
    )
    parser.add_argument("wkt_file", help="Input WKT file.")
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument("--start-date", default="2014-09-01", help="Start date in YYYY-mm-dd format.")
    parser.add_argument("--end-date", default=str(pd.Timestamp.today().date()), help="End date in YYYY-mm-dd format.")

    args = parser.parse_args()

    wkt_file = args.wkt_file
    out_dir = args.out_dir

    logger.info(f"Downloading S1 metadata from WKT file {wkt_file} into {out_dir}")

    biweekly_dates = pd.date_range(start=args.start_date, end=args.end_date, freq="2W").strftime(
        "%Y-%m-%d"
    )

    for start, end in zip(biweekly_dates[:-1], biweekly_dates[1:]):
        logger.info(f"Downloading {start} to {end}")
        try:
            download_with_retry(
                out_dir=out_dir,
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


if __name__ == "__main__":
    main()
