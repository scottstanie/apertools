#!/usr/bin/env python
"""
Script for making small tifs of complex data for quick looks

Uses gdal_translate to reduce size, then gdal_calc to get power from complex values
"""
import itertools
# import multiprocessing
# import os
import numpy as np
import argparse
import rasterio as rio

from apertools import sario

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--outfile",
        default="tiles",
        help="Name of output file (default=%(default)s)",
    )
    p.add_argument(
        "--out-pct",
        type=float,
        default=1,
        help="pct size of original igrams (default=%(default)s)",
    )
    args = p.parse_args()

    # gdal_translate -outsize 2% 2% S1B_20190325.geo.vrt looked_S1B_20190325.geo.tif
    ifg_date_list = sario.find_igrams(".")
    unw_file_list = [f.replace(".int", ".unw") for f in sario.find_igrams(".", parse=False)]

    geo_dates = sorted(set(itertools.chain.from_iterable(ifg_date_list)))
    with rio.open(unw_file_list[0]) as ds:
        out = np.zeros((ds.height, ds.width))
        transform = ds.transform
        crs = ds.crs

    print(out.shape)

    for gdate in geo_dates:
        cur_unws = [f for (pair, f) in zip(ifg_date_list, unw_file_list) if gdate in pair]
        out = 0
        for unwf in cur_unws:
            with rio.open(unwf) as ds:
                # phase band is #2, amplitde is 1
                out += ds.read(2)

        print("Averaging {} unwrapped igrams for {}".format(len(cur_unws), gdate))
        out /= len(cur_unws)
        outfile = "avg_" + gdate.strftime("%Y%m%d") + ".tif"
        with rio.open(
                outfile,
                "w",
                crs=crs,
                transform=transform,
                driver="GTiff",
                height=out.shape[0],
                width=out.shape[1],
                count=1,
                dtype=out.dtype,
        ) as dst:
            dst.write(out, 1)
