#!/usr/bin/env python
"""
Averages all unwrapped igrams, making images of the averge phase per date
"""
import glob
import itertools
import subprocess
import datetime

# import multiprocessing
import os
import h5py
import numpy as np
import argparse
import rasterio as rio
from pathlib import Path

from apertools import sario, latlon, utils


def get_cli_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--slc-vrt-dir",
        default="../",
        help="filename extension of unwrapped igrams to average (default=%(default)s)",
    )
    p.add_argument(
        "--slc-vrt-glob",
        default="*.geo.vrt",
        help="Bash search string to find SLC VRT files (default=%(default)s)",
    )
    p.add_argument(
        "--out-dir",
        "-o",
        default="./raider",
        help="Output location of delays files. (default=%(default)s)",
    )
    p.add_argument(
        "--model",
        choices=[
            "ERA5",
            "ERA5T",
            "ERAI",
            "MERRA2",
            "WRF",
            "HRRR",
            "GMAO",
            "HDF5",
            "HRES",
            "NCMR",
        ],
        default="GMAO",
        type=str.upper,
        help="Name of weather model. (default=%(default)s)",
    )
    p.add_argument(
        "--dem-file",
        default="./elevation_looked.dem",
        help="location of DEM file. (default=%(default)s)",
    )
    p.add_argument(
        "--los-enu-file",
        default="./los_enu.tif",
        help="location of ENU line of sight tif . (default=%(default)s)",
    )
    # p.add_argument(
    #     "--overwrite",
    #     action="store_true",
    #     default=False,
    #     help="Overwrite existing averaged files (default=%(default)s)",
    # )
    return p.parse_args()


def run_raider(dates, time, dem_file, los_file, lat_file, lon_file, model="GMAO", out_dir="./raider"):
    for date in dates:
        cmd = (
            f"raiderDelay.py --model {model}  --time {time} --lineofsight {los_file}"
            f"-d {dem_file} --latlon {lat_file} {lon_file} --out {out_dir} --date {date}"
        )
        print(cmd)
        subprocess.check_call(cmd, shell=True)


def make_latlon_vrts(
    lat, lon, rsc_file, lat_file_bin="lat.geo", lon_file_bin="lon.geo"
):
    dtype = "float32"
    lat.astype(dtype).tofile(lat_file_bin)
    lon.astype(dtype).tofile(lon_file_bin)
    sario.save_vrt(filename=lat_file_bin, rsc_file=rsc_file, num_bands=1, dtype=dtype)
    sario.save_vrt(filename=lon_file_bin, rsc_file=rsc_file, num_bands=1, dtype=dtype)
    return lat_file_bin + ".vrt", lon_file_bin + ".vrt"


if __name__ == "__main__":
    args = get_cli_args()
    Path(args.out_dir).mkdir(exist_ok=True, parents=True)

    rsc_file = f"{args.dem_file}.rsc"
    los_az_inc_file = utils.enu_to_az_inc(args.los_enu_file)
    lat, lon = latlon.grid(**sario.load(rsc_file))
    slc_list = Path(args.slc_vrt_dir).glob(args.slc_vrt_glob)

    dates = []
    for f in slc_list:
        with rio.open(f) as src:
            try:
                dt = datetime.datetime.fromisoformat(src.tags()["acquisition_datetime"])
            except:
                print(f"{f} does not have acquisition_datetime tag. Skipping")
                continue
            time = dt.strftime("%H:%M:%S")
            # Out[28]: '00:59:12'
            dates.append(dt.strftime("%Y%m%d"))

    lat_file, lon_file = make_latlon_vrts(lat, lon, rsc_file)
    print(f"Running raider for {len(dates)} dates")

    run_raider(
        dates,
        time,
        args.dem_file,
        los_az_inc_file,
        lat_file,
        lon_file,
        model=args.model,
        out_dir=args.out_dir,
    )
