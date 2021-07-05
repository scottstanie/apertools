#!/usr/bin/env python
"""
Averages all unwrapped igrams, making images of the averge phase per date
"""
import glob
import itertools
import subprocess

# import multiprocessing
import os
import h5py
import numpy as np
import argparse
import rasterio as rio

from apertools import sario
from apertools.prepare import remove_ramp


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
    p.add_argument(
        "--search-path",
        "-p",
        default=".",
        help="location of igram files. (default=%(default)s)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing averaged files (default=%(default)s)",
    )
    p.add_argument(
        "--normalize-time",
        "-n",
        action="store_true",
        default=False,
        help="Divide igram phase by temporal baseline (default=%(default)s)",
    )
    return p.parse_args()


def create_averages(
    deramp, ext, search_path=".", overwrite=False, normalize_time=False
):

    # gdal_translate -outsize 2% 2% S1B_20190325.geo.vrt looked_S1B_20190325.geo.tif
    ifg_date_list = sario.find_igrams(directory=search_path, ext=ext)
    unw_file_list = sario.find_igrams(directory=search_path, ext=ext, parse=False)

    geo_date_list = sorted(set(itertools.chain.from_iterable(ifg_date_list)))
    geo_date_list, ifg_date_list = sario.ignore_geo_dates(
        geo_date_list,
        ifg_date_list,
        ignore_file=os.path.join(search_path, "slclist_ignore.txt"),
    )
    with rio.open(unw_file_list[0]) as ds:
        out = np.zeros((ds.height, ds.width))
        transform = ds.transform
        crs = ds.crs

    mask_fname = os.path.join(search_path, "masks.h5")
    with h5py.File(mask_fname, "r") as f:
        mask_stack = f["igram"][:].astype(bool)

    out_mask = np.zeros_like(out).astype(bool)

    # Get masks for deramping
    mask_igram_date_list = sario.load_ifglist_from_h5(mask_fname)

    for (idx, gdate) in enumerate(geo_date_list):
        outfile = "avg_" + gdate.strftime("%Y%m%d") + ".tif"
        if os.path.exists(outfile) and not overwrite:
            print(f"{outfile} exists: skipping")
            continue

        cur_unws = [
            (f, date_pair)
            for (date_pair, f) in zip(ifg_date_list, unw_file_list)
            if gdate in date_pair
        ]
        # reset the matrix to all zeros
        out = 0
        for unwf, date_pair in cur_unws:
            with rio.open(unwf) as ds:
                # phase band is #2, amplitde is 1
                img = ds.read(2)
                if normalize_time:
                    img /= (date_pair[1] - date_pair[0]).days
                out += img

            mask_idx = mask_igram_date_list.index(date_pair)
            out_mask |= mask_stack[mask_idx]

        print(
            "Averaging {} unwrapped igrams for {} ({} out of {})".format(
                len(cur_unws), gdate, idx + 1, len(geo_date_list)
            )
        )
        out /= len(cur_unws)

        if args.deramp:
            out = remove_ramp(out, deramp_order=1, mask=out_mask)
        else:
            out[out_mask] = np.nan

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
    create_averages(
        args.deramp,
        args.ext,
        search_path=args.search_path,
        overwrite=args.overwrite,
    )
