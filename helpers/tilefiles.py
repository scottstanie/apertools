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
import apertools.sario
import rasterio as rio
import glob


def _translate(vf, out, new_width, new_height, outformat):
    cmd = "gdal_translate -quiet -outsize {w} {h} {inp} {out} -of {of}".format(
        w=new_width,
        h=new_height,
        inp=vf,
        out=out,
        of=outformat,
    )
    print("Running ", cmd)
    subprocess.run(cmd, shell=True)


def convert(vrtlist, new_width, new_height, outformat="ROI_PAC", outext=".slc"):
    outfilelist = ["small_" + f.replace(".geo.vrt", outext) for f in vrtlist]
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
        pool.apply_async(_translate, args=(vf, out, new_width, new_height, outformat))
        for (vf, out) in zip(vleft, oleft)
    ]
    [res.get() for res in results]
    return outfilelist


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
        "--num-tiles-across",
        type=int,
        help="Number of tiles across for display (default = approx square)",
    )
    p.add_argument(
        "--rsc-file",
        default="elevation.dem.rsc",
        help="File with geo transform information (default=%(default)s)",
    )
    args = p.parse_args()

    # First, make sure we've save all the binarys as vrts
    [
        apertools.sario.save_vrt(filename=f, rsc_file=args.rsc_file)
        for f in glob.glob("*.geo")
    ]
    # Also, add .rsc files for them for after we've saved new ones with the gdal driver
    [
        apertools.utils.force_symlink(args.rsc_file, f + ".rsc")
        for f in glob.glob("*.geo")
    ]

    vrtlist = sorted(glob.glob("*.geo.vrt"))
    # gdal_translate -outsize 2% 2% S1B_20190325.geo.vrt looked_S1B_20190325.geo.tif

    num_tiles_across = int(floor(sqrt(len(vrtlist))))

    with rio.open(vrtlist[0]) as ds:
        orig_width, orig_height = ds.width, ds.height
    factor = int(100 / args.out_pct)
    new_width, new_height = orig_width // factor, orig_height // factor

    # ROI_PAC slc format
    outfilelist = convert(
        vrtlist, new_width, new_height, outformat="ROI_PAC", outext=".slc"
    )

    # Now convert the small binary slcs into one tile file
    stackfile = "stackfile"
    # usage: makestackfile filelist stackfile length nigrams
    cmd = f"makestackfile filelist.txt {stackfile} {new_width} {len(outfilelist)}"
    print("Running ", cmd)
    subprocess.run(cmd, shell=True)
    # usage: tilefile infile outfile tiles_across len lines
    cmd = "tilefile {inp} {out} {ta} {w} {h}".format(
        inp=stackfile,
        out=args.outfile,
        ta=num_tiles_across,
        w=new_width,
        h=new_height,
    )
    print("Running ", cmd)
    subprocess.run(cmd, shell=True)
    print("View tilefile with following command:")
    print("dismph {} {}".format(args.outfile, num_tiles_across * new_width))
