#!/usr/bin/env python
"""
Averages all unwrapped igrams, making images of the averge phase per date
"""
import glob
import itertools
import subprocess

# import multiprocessing
# import os
import numpy as np
import argparse
import rasterio as rio

from apertools import sario
from insar.prepare import remove_ramp


def get_cli_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--deramp",
        action="store_true",
        default=True,
        help="remove a linear ramp from phase after averaging (default=%(default)s)",
    )
    p.add_argument(
        "--ext",
        default=".unw",
        help="filename extension of unwrapped igrams to average (default=%(default)s)",
    )
    return p.parse_args()


def create_averages(deramp, ext):

    # gdal_translate -outsize 2% 2% S1B_20190325.geo.vrt looked_S1B_20190325.geo.tif
    ifg_date_list = sario.find_igrams(directory=".", ext=args.ext)
    unw_file_list = sario.find_igrams(".", ext=args.ext, parse=False)
    # unw_file_list = [f.replace(".int", ".unwflat") for f in sario.find_igrams(".", parse=False)]

    geo_dates = sorted(set(itertools.chain.from_iterable(ifg_date_list)))
    with rio.open(unw_file_list[0]) as ds:
        out = np.zeros((ds.height, ds.width))
        transform = ds.transform
        crs = ds.crs

    for (idx, gdate) in enumerate(geo_dates):
        cur_unws = [
            f for (pair, f) in zip(ifg_date_list, unw_file_list) if gdate in pair
        ]
        # reset the matrix to all zeros
        out = 0
        for unwf in cur_unws:
            with rio.open(unwf) as ds:
                # phase band is #2, amplitde is 1
                out += ds.read(2)

        print(
            "Averaging {} unwrapped igrams for {} ({} out of {})".format(
                len(cur_unws), gdate, idx + 1, len(geo_dates)
            )
        )
        out /= len(cur_unws)

        if args.deramp:
            out = remove_ramp(out, deramp_order=1, mask=np.ma.nomask)

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
            nodata=0,
            dtype=out.dtype,
        ) as dst:
            dst.write(out, 1)
        # And make it easily viewable as a Byte image
        subprocess.call(
            [
                "gdal_translate",
                "-quiet",
                "-scale",
                "-co",
                "TILED=YES",
                "-co",
                "COMPRESS=LZW",
                "-ot",
                "Byte",
                outfile,
                outfile.replace(".tif", "_byte.tif"),
            ]
        )
        # if idx > 3:
        #     break


def load_avg_igrams():
    avgs = []
    fnames = glob.glob("avg*[0-9].tif")
    for f in fnames:
        with rio.open(f) as ds:
            avgs.append(ds.read(1))

    avgs = np.stack(avgs)
    return avgs, fnames


def plot_avgs(avgs=None, fnames=None, cmap="seismic"):
    import matplotlib.pyplot as plt

    if avgs is None or fnames is None:
        avgs, fnames = load_avg_igrams()

    vmin, vmax = np.nanmin(avgs), np.nanmax(avgs)
    vm = np.max(np.abs([vmin, vmax]))
    ntiles = int(np.ceil(np.sqrt(len(avgs))))
    fig, axes = plt.subplots(ntiles, ntiles)
    for (avg, ax, fn) in zip(avgs, axes.ravel(), fnames):
        axim = ax.imshow(avg, vmin=-vm, vmax=vm, cmap=cmap)
        ax.set_title(f"{fn}: {np.var(avg):.2f}")
        fig.colorbar(axim, ax=ax)
    return fig, axes


if __name__ == "__main__":
    args = get_cli_args()
    create_averages(args.deramp, args.ext)
