#!/usr/bin/env python
import argparse
import datetime
import glob
import os
import subprocess

import numpy as np
import shapely.geometry
import rasterio as rio
import pandas as pd
import geopandas as gpd

import coseismic_stack
import apertools.sario as sario
import apertools.latlon as latlon
import apertools.utils as utils
import apertools.subset as subset

from insar.prepare import remove_ramp
from insar.timeseries import PHASE_TO_CM

# TODO: merge overlapping date/areas... aftershocks means 3/26,27,29 all there

TEXNET_DATA = "/home/scott/cisr-data/texnet_events_20201012.csv"


def run_vrt_subset(
    igram_dir,
    geo_dir,
    mag_thresh=3,
    eq_fname=TEXNET_DATA,
    sar_fname="../S1A_20141215.geo.vrt",
    box_size=20,
    ignore_geos=True,
    num_igrams=10,
    igram_type="cross",
    verbose=True,
):
    df = setup_folders(eq_fname=eq_fname, sar_fname=sar_fname, mag_thresh=mag_thresh)
    all_vrts = subset_all_folders(
        df,
        igram_dir,
        geo_dir,
        ignore_geos=ignore_geos,
        num_igrams=num_igrams,
        igram_type=igram_type,
        verbose=verbose,
    )
    plot_stacks(all_vrts)


def setup_folders(
    eq_fname=TEXNET_DATA,
    sar_fname="../S1A_20141215.geo.vrt",
    mag_thresh=3,
    box_size=20,
):
    df = load_eq_df(
        eq_fname=eq_fname, sar_fname=sar_fname, mag_thresh=mag_thresh, box_size=box_size
    )
    for (index, row) in df.iterrows():
        print(f"Creating directory: {row.dirname}")
        os.makedirs(row.dirname, exist_ok=True)
    return df


def load_eq_df(
    eq_fname=TEXNET_DATA,
    mag_thresh=3,
    box_size=20,
    sar_fname="../S1A_20141215.geo.vrt",
):
    """Create a DataFrame of the top weekly earthquakes, limited to the
    area within sar_fname and above magnitude mag_thresh
    """
    eqdf = read_eqs(eq_fname)
    df = get_top_mag_dates(eqdf)
    if sar_fname is not None:
        df = get_eqs_in_bounds(sar_fname, df, mag_thresh=mag_thresh)
    df.index = pd.to_datetime(df.index)

    df["bbox"] = df.geometry.buffer(latlon.km_to_deg(box_size))
    # df["dirname"] = df.index.strftime("%Y%m%d").str.cat(df.event_id, sep="_")
    df["dirname"] = eq_dirname(df)
    return df


def eq_dirname(df):
    names = []
    for row in df.itertuples():
        dt = row.Index.strftime("%Y%m%d")
        ll = utils.pprint_lon_lat(row.lon, row.lat)
        names.append(f"{dt}_{ll}_{row.event_id}")
    return names


def subset_all_folders(df, igram_dir, geo_dir, **kwargs):
    all_vrts = []
    for event_date, row in df.iterrows():
        try:
            vrts = subset_unws(event_date, row, igram_dir, geo_dir, **kwargs)
            all_vrts.append(vrts)
        except Exception as e:
            print(f"Failed on {event_date}: {e}")
            continue

    return all_vrts


def plot_stacks(all_vrts):
    import matplotlib.pyplot as plt

    nimgs = len(all_vrts)
    ntiles = int(np.ceil(np.sqrt(nimgs)))
    fig, axes = plt.subplots(ntiles, ntiles)
    for (aa, ax) in zip(all_vrts, axes.ravel()):
        p = stack_vrts(aa)
        axim = ax.imshow(p, vmax=1, vmin=-1, cmap="seismic_wide_y")

        print(f"averaged {len(aa)} igrams, done with {aa[0]}...")
        ax.set_title(aa[0].split("/")[-2])

    fig.colorbar(axim, ax=axes.ravel()[nimgs - 1])


# TODO: refactor coseismic_stack functions to not dupe this code
def stack_vrts(
    stack_fnames,
    ref=(5, 5),
    window=5,
):
    phase_subset_stack = []
    for f in stack_fnames:
        with rio.open(f) as src:
            cur = src.read(2)
        deramped_phase = remove_ramp(np.squeeze(cur), deramp_order=1, mask=np.ma.nomask)
        phase_subset_stack.append(deramped_phase)

    phase_subset_stack = np.mean(np.stack(phase_subset_stack, axis=0), axis=0)
    # subtract the reference location:
    ref_row, ref_col = ref
    win = window // 2
    patch = phase_subset_stack[
        ref_row - win : ref_row + win + 1, ref_col - win : ref_col + win + 1
    ]
    phase_subset_stack -= np.nanmean(patch)
    phase_subset_stack *= PHASE_TO_CM
    return phase_subset_stack


def subset_unws(
    event_date,
    row,
    igram_dir,
    geo_dir,
    ignore_geos=True,
    num_igrams=10,
    igram_type="cross",
    verbose=True,
):
    """
    event_date is from index of df
    row has columns: max_mag,lat,lon,event_id,geometry,bbox,dirname
    """
    gi_file = "geolist_ignore.txt" if ignore_geos else None
    geolist, intlist = sario.load_geolist_intlist(
        igram_dir, geo_dir=geo_dir, geolist_ignore_file=gi_file
    )
    if igram_type == "cross":
        stack_igrams = coseismic_stack.select_cross_event(
            geolist, intlist, event_date, num_igrams=num_igrams
        )
    elif igram_type == "pre":
        min_date = event_date - datetime.timedelta(days=180)
        stack_igrams = coseismic_stack.select_pre_event(
            geolist, intlist, event_date, min_date=min_date
        )
    elif igram_type == "post":
        max_date = event_date + datetime.timedelta(days=180)
        stack_igrams = coseismic_stack.select_post_event(
            geolist, intlist, event_date, max_date=max_date
        )
    else:
        raise ValueError("igram_type must be 'cross', 'pre', 'post")

    stack_fnames = sario.intlist_to_filenames(stack_igrams, ".unw")
    stack_fullpaths = [
        os.path.join(os.path.abspath(igram_dir), f) for f in stack_fnames
    ]
    vrt_fnames = [
        os.path.join(os.path.abspath(row.dirname), f + ".vrt") for f in stack_fnames
    ]
    if verbose:
        print("Using the following igrams in stack:")
        for (fin, fout) in zip(stack_fullpaths, vrt_fnames):
            print(fout, "->", fin)

    for (in_fname, out_fname) in zip(stack_fullpaths, vrt_fnames):
        subset.copy_vrt(
            in_fname, out_fname=out_fname, bbox=row.bbox.bounds, verbose=verbose
        )

    return vrt_fnames


def read_eqs(fname=TEXNET_DATA):
    df = pd.read_csv(
        fname,
        usecols=(
            "EventID",
            "Origin Date",
            "Origin Time",
            "Latitude (WGS84)",
            "Longitude (WGS84)",
            "Magnitude",
            "Depth of Hypocenter (Km.  Rel to MSL)",
        ),
        parse_dates=[["Origin Date", "Origin Time"]],
    )
    df.columns = ["dt", "event_id", "mag", "lat", "lon", "depth"]
    df = df.set_index("dt")
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))


def get_top_mag_dates(df, n=None):
    dailydf = df.copy()
    # dailydf.reset_index(drop=True, inplace=True)  # turns into RangeIndex
    dailydf["date"] = dailydf.index.date

    # dailydf = dailydf.groupby("date")["mag"].agg(max_mag="max", idxmax="idxmax")
    weeklydf = (
        dailydf.resample("W").agg({"mag": ["max", "idxmax"]}).droplevel(0, axis=1)
    )
    weeklydf.rename(columns={"max": "max_mag"}, inplace=True)

    weeklydf[["lat", "lon", "depth", "event_id"]] = df.loc[
        weeklydf["idxmax"], ["lat", "lon", "depth", "event_id"]
    ].values
    weeklydf = (
        weeklydf[["max_mag", "lat", "lon", "depth", "event_id"]]
        .sort_values("max_mag", ascending=False)
        .head(n)
    )
    return gpd.GeoDataFrame(
        weeklydf, geometry=gpd.points_from_xy(weeklydf.lon, weeklydf.lat)
    )


# Getting counties within texas shape
# from geopandas.tools import sjoin
# counties = gpd.read_file("countyl010g_shp_nt00964/countyl010g.shp")
# states = gpd.read_file("state_boundaries/cb_2018_us_state_5m.shp")
# texas = states[states.NAME == 'Texas']
# texas_counties = sjoin(counties, texas, how='inner', op='within')
# gdf = gpd.GeoDataFrame(topdf, geometry=gpd.points_from_xy(topdf.lon, topdf.lat))


def get_eqs_in_bounds(insar_fname, eqdf, mag_thresh=3):
    with rio.open(insar_fname) as src:
        idxs = eqdf.within(shapely.geometry.box(*src.bounds))
        outdf = eqdf[idxs]
    return outdf[outdf.max_mag > mag_thresh] if mag_thresh else outdf


### PART 2
# Downloading new stuff


def download_and_process(df, xrate=7, yrate=2, looks=3):
    from apertools import asfdownload
    from apertools import createdem

    #                     event_id  mag lat lon depth geometry bbox
    # dt
    # 2018-10-20 13:04:31  texnet2018uomw  4.2  35.3549 -101.7199 ...
    # 1: get bbox for DEM generation
    bbox = df.iloc[0].bbox.bounds
    dem_fname = "elevation.dem"
    if not os.path.exists(dem_fname):
        createdem.main(*bbox, xrate=xrate, yrate=yrate, outname=dem_fname)

    # 2: get download time range
    time_pad = datetime.timedelta(days=180)
    event_time = df.index[0]
    start, end = event_time - time_pad, event_time + time_pad
    query_fname = asfdownload.query_only(
        start=start, end=end, dem=dem_fname, query_filetype="geojson"
    )
    path_nums, starts = asfdownload.parse_query_results(query_fname)
    new_dirs = []
    for path_num, direction in path_nums.keys():
        dirname = f"path_{path_num}_{direction}"
        utils.mkdir_p(dirname)
        new_dirs.append(dirname)

    for dirname in new_dirs:
        print(f"Downloading data for {dirname}")
        asfdownload.download_data(start=start, end=end, dem=dem_fname, out_dir=dirname)
        for f in glob.glob(dem_fname + "*"):
            utils.force_symlink(f, os.path.join(dirname, f))

    process_dirs(new_dirs, xrate, yrate, looks)


def process_dirs(new_dirs, xrate, yrate, looks, ref_row=5, ref_col=5):
    # Process results
    xlooks = looks * xrate
    ylooks = looks * yrate

    for dirname in new_dirs:
        os.chdir(dirname)
        cmd = (
            f"insar process --start 3 --xlooks {xlooks} --ylooks {ylooks} "
            f" --ref-row {ref_row} --ref-col {ref_col} --gpu"
        )
        subprocess.check_call(cmd, shell=True)
        os.chdir("..")


def get_cli_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--igram-dir",
        default=".",
        help="location of igram files. (default=%(default)s)",
    )
    p.add_argument(
        "--geo-dir",
        default="..",
        help="location of .geo SLC files. (default=%(default)s)",
    )
    p.add_argument(
        "--mag-thresh",
        default=3,
        help="Magnitude of earthquakes to threshold (default=%(default)s)",
    )
    p.add_argument(
        "--sar-fname",
        help="SLC filename of TexNet earthquake data (default=%(default)s)",
    )
    p.add_argument(
        "--box-size",
        default=20,
        help="Size (in km) of box to subset around EQ hypocenter (default=%(default)s)",
    )
    p.add_argument(
        "--eq-fname",
        default=TEXNET_DATA,
        help="filename of TexNet earthquake data (default=%(default)s)",
    )
    p.add_argument(
        "--ignore-geos",
        action="store_true",
        default=True,
        help="Apply the geolist_ignore.txt file (default=%(default)s)",
    )
    return p.parse_args()


if __name__ == "__main__":
    print("ok")
    args = get_cli_args()
    run_vrt_subset(
        args.igram_dir,
        args.geo_dir,
        mag_thresh=args.mag_thresh,
        eq_fname=args.eq_fname,
        sar_fname=args.sar_fname,
        box_size=args.box_size,
        ignore_geos=args.ignore_geos,
        num_igrams=10,
        igram_type="cross",
        verbose=True,
    )
