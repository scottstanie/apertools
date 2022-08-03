#!/usr/bin/env python
from logging import getLogger, StreamHandler, DEBUG
from asyncio.log import logger
import sys

import pandas as pd

from apertools import asfdownload

logger = getLogger(__name__)

def main():
    wkt_file = sys.argv[1]
    today = str(pd.Timestamp.today().date())
    biweekly_dates =  pd.date_range(start="2014-01-01", end=today, freq='2W').strftime("%Y-%m-%d")

    for start, end in zip(biweekly_dates[:-1], biweekly_dates[1:]):
        logger.info(f"Downloading {start} to {end}")
        asfdownload.form_url(
            wkt_file=wkt_file,
            start=start,
            end=end,
            processingLevel="METADATA_SLC",
            maxResults=2000,
            platform="S1",
            beamMode="IW",
        )


if __name__ == "__main__":
    main()
