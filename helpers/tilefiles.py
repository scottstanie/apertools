#!/usr/bin/env python
"""
Script for making small tifs of complex data for quick looks

Uses gdal_translate to reduce size, then gdal_calc to get power from complex values
"""
import multiprocessing
from math import floor, sqrt
import os
import argparse
import subprocess
import rasterio as rio
import glob


def _translate(vf, out, new_width, new_height):
    cmd = "gdal_translate -quiet -outsize {w} {h} {inp} {out} -of ROI_PAC".format(
        w=new_width,
        h=new_height,
        inp=vf,
        out=out,
    )
    print("Running ", cmd)
    subprocess.run(cmd, shell=True)


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
    p.add_argument(
        "--num_tiles_across",
        type=int,
        help="Number of tiles across for display (default = approx square)",
    )
    args = p.parse_args()

    # gdal_translate -outsize 2% 2% S1B_20190325.geo.vrt looked_S1B_20190325.geo.tif
    vrtlist = glob.glob("*.geo.vrt")

    num_tiles_across = int(floor(sqrt(len(vrtlist))))

    with rio.open(vrtlist[0]) as ds:
        orig_width, orig_height = ds.width, ds.height
    factor = int(100 / args.out_pct)
    new_width, new_height = orig_width // factor, orig_height // factor

    outfilelist = [f.replace(".geo.vrt", ".slc") for f in vrtlist]  # ROI_PAC slc format
    with open("filelist.txt", "w") as fl:
        fl.write("\n".join(outfilelist) + "\n")

    # TODO: check remaining files not existing?
    vleft, oleft = [], []
    for (vf, out) in zip(vrtlist, outfilelist):
        if not os.path.exists(out):
            vleft.append(vf)
            oleft.append(out)

    max_jobs = 4
    pool = multiprocessing.Pool(max_jobs)
    results = [
        pool.apply_async(_translate, args=(vf, out, new_width, new_height))
        for (vf, out) in zip(vleft, oleft)
    ]
    [res.get() for res in results]

    stack_file = "stackfile"
    # usage: makestackfile filelist stackfile length nigrams
    cmd = "makestackfile filelist.txt {stack} {w} {N}".format(
        stack=stack_file,
        w=new_width,
        N=len(outfilelist),
    )
    print("Running ", cmd)
    subprocess.run(cmd, shell=True)
    # usage: tilefile infile outfile tiles_across len lines
    cmd = "tilefile {inp} {out} {ta} {w} {h}".format(
        inp=stack_file,
        out=args.outfile,
        ta=num_tiles_across,
        w=new_width,
        h=new_height,
    )
    print("Running ", cmd)
    subprocess.run(cmd, shell=True)
    print("View tilefile with following command:")
    print("dismph {} {}".format(args.outfile, num_tiles_across * new_width))
