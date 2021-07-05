#!/usr/bin/env python
import argparse
import datetime
import glob
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry
import rasterio as rio
import pandas as pd
import geopandas as gpd

import apertools.coseismic_stack as coseismic_stack
import apertools.sario as sario
import apertools.latlon as latlon
import apertools.utils as utils
import apertools.subset as subset

from apertools.deramp import remove_ramp
from apertools.constants import PHASE_TO_CM

# TODO: merge overlapping date/areas... aftershocks means 3/26,27,29 all there

TEXNET_DATA = "/home/scott/cisr-data/texnet_events_20201012.csv"


def run_vrt_subset(
    igram_dir,
    geo_dir,
    mag_thresh=3,
    eq_fname=TEXNET_DATA,
    sar_fname=None,
    box_size=20,
    remove_outliers=False,
    ignore_geos=True,
    ref=(5, 5),
    num_igrams=10,
    igram_type="cross",
    verbose=True,
    outname="stackavg.tif",
    plot_block=True,
):
    if sar_fname is None:
        sar_fname = glob.glob(os.path.join(geo_dir, "*.geo.vrt"))[0]
        print(f"Using bbox area {sar_fname = } to bound search for EQs.")
    df = setup_folders(
        eq_fname=eq_fname, sar_fname=sar_fname, mag_thresh=mag_thresh, box_size=box_size
    )
    if len(df) < 1:
        return
    all_vrts = subset_all_folders(
        df,
        igram_dir,
        geo_dir,
        remove_outliers=remove_outliers,
        ignore_geos=ignore_geos,
        num_igrams=num_igrams,
        igram_type=igram_type,
        verbose=verbose,
    )
    imgs = []
    for vrt_group in all_vrts:
        print(vrt_group)
        img = calculate_stack(vrt_group, ref=ref, outname=outname, overwrite=True)
        if img is not None:
            imgs.append(img)
    vrt_names = get_vrt_names(all_vrts)
    plot_stacks(imgs, vrt_names, block=plot_block)


def setup_folders(
    eq_fname=TEXNET_DATA,
    geo_dir=None,
    sar_fname=None,
    mag_thresh=3,
    box_size=20,
):
    if geo_dir is not None:
        sar_fname = glob.glob(os.path.join(geo_dir, "*.geo.vrt"))[0]
        print(f"Using bbox area {sar_fname = } to bound search for EQs.")

    df = load_eq_df(
        eq_fname=eq_fname, sar_fname=sar_fname, mag_thresh=mag_thresh, box_size=box_size
    )
    print(f"Found {df.shape[0]} earthquake rows")
    for (index, row) in df.iterrows():
        print(f"Creating directory: {row.dirname}")
        os.makedirs(row.dirname, exist_ok=True)
    return df


def load_eq_df(
    eq_fname=TEXNET_DATA,
    mag_thresh=3,
    box_size=20,
    sar_fname=None,
    decimals=1,  # How many lat/lon decimals should the directories have?
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
    df["dirname"] = eq_dirname(df, decimals=decimals)
    return df


def eq_dirname(df, decimals=1):
    names = []
    # row looks like:
    # Pandas(Index=Timestamp('2020-03-29 00:00:00'), max_mag=4.6, lat=31.7025, lon=-104.0521,
    # depth=6.2, event_id='texnet2020galz', geometry=..., bbox=..)
    for row in df.itertuples():
        dt = row.Index.strftime("%Y%m%d")
        ll = utils.pprint_lon_lat(row.lon, row.lat, decimals=decimals)
        names.append(f"{dt}_{ll}_mag{row.max_mag:.1f}_{row.event_id}")
    return names


def calculate_stack(vrt_group, outname="stackavg.tif", ref=(5, 5), overwrite=False):
    if len(vrt_group) < 2:
        # TODO: no name for empty... maybe pass dict {folder: [unws...]}
        print(f"{len(vrt_group)} igrams for {vrt_group}, skipping...")
        return None
    directory = os.path.dirname(vrt_group[0])
    if outname:
        out_filename = os.path.join(directory, outname)
        if os.path.exists(out_filename) and not overwrite:
            print(f"{out_filename} exists and {overwrite = }, loading...")
            with rio.open(out_filename) as src:
                return src.read(1)

    avg_velo = stack_vrts(vrt_group, ref=ref)

    print(f"averaged {len(vrt_group)} igrams in {directory}...")
    if outname:
        print(f"Saving to {out_filename}")
        sario.save_image_like(avg_velo, out_filename, vrt_group[0], driver="GTiff")
    return avg_velo


def get_vrt_names(all_vrts):
    return [aa[0].split("/")[-2] for aa in all_vrts]


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


# TODO: merge this with "avg_igram" script


def calc_avg_geos(
    event_date,
    igram_dir,
    geo_dir,
    bbox,
    num_igrams=10,
    outdir="average_geos",
    ext=".unw",
    overwrite=False,
):
    stack_igrams, _ = select_subset_igrams(
        event_date, igram_dir, geo_dir, num_igrams=num_igrams
    )
    geo_date_list = utils.slclist_from_igrams(stack_igrams)
    fully_connected_igrams = utils.full_igram_list(geo_date_list)
    src_fullpaths = [
        os.path.join(os.path.abspath(igram_dir), f)
        for f in sario.ifglist_to_filenames(fully_connected_igrams, ext)
    ]

    print(f"Creating directory: {outdir}")
    os.makedirs(outdir, exist_ok=True)
    vrt_filenames = subset_unws(src_fullpaths, outdir, bbox, verbose=False)

    geo_to_igram_dict = {
        geo: [pair for pair in fully_connected_igrams if geo in pair]
        for geo in geo_date_list
    }
    vrt_ext = ext + ".vrt"  # TODO: extract from vrt_filenames?
    geo_to_fname_dict = {
        geo: [
            os.path.join(outdir, f)
            for f in sario.ifglist_to_filenames(ig_list, vrt_ext)
        ]
        for (geo, ig_list) in geo_to_igram_dict.items()
    }
    avg_imgs = []
    for (geo, fname_list) in geo_to_fname_dict.items():
        outname = f"avg_{geo.strftime('%Y%m%d')}.tif"
        img = calculate_stack(fname_list, outname=outname, overwrite=False)
        if img is not None:
            avg_imgs.append(img)

    return geo_date_list, np.array(avg_imgs)


def find_geo_outliers(geo_date_list, avg_imgs, nsigma=3):
    """Using the average of interferograms per date, find especially noisy ones

    Method: for N average SAR images...
        1. find the overall variance of each image (N total numbers)
        2. find the median of the N points
        3. set a cutoff of `median + 3sigma`
    Returns the dates from `geo_date_list` which are flagged to be outliers
    """
    img_vars = np.var(avg_imgs, axis=(1, 2))
    cutoff = np.median(img_vars) + nsigma * np.std(img_vars)
    outlier_idxs = img_vars > cutoff
    return np.array(geo_date_list)[outlier_idxs].tolist()


def run_jackknife(
    event_date,
    igram_dir,
    geo_dir,
    bbox,
    outdir="jackknife",
    ext=".unw",
    extra_geo_ignores=[],
    plot_std=True,
):
    stack_igrams, _ = select_subset_igrams(event_date, igram_dir, geo_dir)
    geos = utils.slclist_from_igrams(stack_igrams)
    imgs = []
    leave_out_dates = []
    for idx, g in enumerate(geos):
        stack_igrams, src_fullpaths = select_subset_igrams(
            event_date,
            igram_dir,
            geo_dir,
            extra_geo_ignores=extra_geo_ignores + [g],
        )
        od = f"outdir{idx}"
        utils.mkdir_p(od)
        vrts = subset_unws(src_fullpaths, od, bbox, verbose=False)
        img = calculate_stack(vrts, outname=None)
        imgs.append(img)
        leave_out_dates.append(g)
    if plot_std:
        plot_std_img(imgs)
    return imgs, leave_out_dates


def plot_std_img(imgs):
    fig, ax = plt.subplots()
    std_img = np.std(np.array(imgs), axis=0)
    axim = ax.imshow(std_img, cmap="Reds")
    fig.colorbar(axim)
    ax.set_title("Std. dev. of estimates from jackknife")
    return fig, ax, std_img


def subset_all_folders(
    df, igram_dir, geo_dir, num_igrams=10, remove_outliers=True, **kwargs
):
    all_vrts = []
    print(df)
    for event_date, row in df.iterrows():
        bbox = row.bbox.bounds
        current_dir = row.dirname
        try:
            # Get the list of very noisy days to skip:
            if remove_outliers:
                print("Removing outlier SAR dates")
                geo_date_list, avg_imgs = calc_avg_geos(
                    event_date,
                    igram_dir,
                    geo_dir,
                    bbox,
                    num_igrams=num_igrams,
                    outdir=current_dir,
                    overwrite=False,
                )
                bad_dates = find_geo_outliers(geo_date_list, avg_imgs, nsigma=3)
                print(f"Found {len(bad_dates)} bad SAR date: {bad_dates}")
            else:
                bad_dates = []

            stack_igrams, src_fullpaths = select_subset_igrams(
                event_date,
                igram_dir,
                geo_dir,
                extra_geo_ignores=bad_dates,
                num_igrams=num_igrams,
                **kwargs,
            )
            vrts = subset_unws(src_fullpaths, current_dir, bbox, **kwargs)
            all_vrts.append(vrts)
        except Exception as e:
            print(f"Failed on {event_date}: {e}")
            continue

    return all_vrts


def select_subset_igrams(
    event_date,
    igram_dir,
    geo_dir,
    ignore_geos=True,
    extra_geo_ignores=None,
    num_igrams=10,
    igram_type="cross",
    ext=".unw",
    **kwargs,
):
    """Pick the igrams to use for the stack, and form full filepaths """
    gi_file = "slclist_ignore.txt" if ignore_geos else None
    slclist, ifglist = sario.load_slclist_ifglist(
        igram_dir, geo_dir=geo_dir, slclist_ignore_file=gi_file
    )
    if extra_geo_ignores is not None:
        slclist = [g for g in slclist if g not in extra_geo_ignores]
        ifglist = [
            pair for pair in ifglist if all(g not in pair for g in extra_geo_ignores)
        ]
    if igram_type == "cross":
        stack_igrams = coseismic_stack.select_cross_event(
            slclist, ifglist, event_date, num_igrams=num_igrams
        )
    elif igram_type == "pre":
        min_date = event_date - datetime.timedelta(days=180)
        stack_igrams = coseismic_stack.select_pre_event(
            slclist, ifglist, event_date, min_date=min_date
        )
    elif igram_type == "post":
        max_date = event_date + datetime.timedelta(days=180)
        stack_igrams = coseismic_stack.select_post_event(
            slclist, ifglist, event_date, max_date=max_date
        )
    else:
        raise ValueError("igram_type must be 'cross', 'pre', 'post")
    stack_fnames = sario.ifglist_to_filenames(stack_igrams, ext)
    src_fullpaths = [os.path.join(os.path.abspath(igram_dir), f) for f in stack_fnames]
    return stack_igrams, src_fullpaths


def subset_unws(
    src_fullpaths,
    out_dirname,
    bbox,
    verbose=True,
    **kwargs,
):
    """
    row has columns: max_mag,lat,lon,event_id,geometry,bbox,dirname
    """
    stack_fnames = [os.path.split(f)[1] for f in src_fullpaths]

    vrt_fnames = [
        os.path.join(os.path.abspath(out_dirname), f + ".vrt") for f in stack_fnames
    ]
    if verbose:
        print("Using the following igrams in stack:")
        for (fin, fout) in zip(src_fullpaths, vrt_fnames):
            print(fout, "->", fin)

    for (in_fname, out_fname) in zip(src_fullpaths, vrt_fnames):
        subset.copy_vrt(in_fname, out_fname=out_fname, bbox=bbox, verbose=verbose)

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


def plot_stacks(imgs, vrt_names, block=False, show_colorbar=True, hide_axes=True):

    nimgs = len(imgs)
    nrow, ncol = _plot_rows_cols(len(imgs))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False)
    for (name, ax, img) in zip(vrt_names, axes.ravel(), imgs):
        axim = ax.imshow(img, vmax=1, vmin=-1, cmap="seismic_wide_y")
        ax.set_title(name)
        if hide_axes:
            ax.set_axis_off()

    if show_colorbar:
        fig.colorbar(axim, ax=axes.ravel()[nimgs - 1])

    plt.show(block=block)


def plot_avg(
    directory=None,
    prefix="avg_",
    band=1,
    cmap="seismic_wide_y",
    hide_axes=True,
    num_igrams=-1,
):
    filenames = sorted(glob.glob(os.path.join(directory, f"{prefix}*")))[:num_igrams]
    print(f"Found {len(filenames)} average images in {directory}")
    imgs = np.stack([sario.load(f, band=band) for f in filenames], axis=0)
    vmax = np.max(np.abs(imgs))
    vmin = -vmax

    nrow, ncol = _plot_rows_cols(len(imgs))
    fig, axes = plt.subplots(nrow, ncol)
    for (img, ax, fname) in zip(imgs, axes.ravel(), filenames):
        axim = ax.imshow(img, vmax=vmax, vmin=vmin, cmap=cmap)
        title = os.path.split(fname)[1]
        title = os.path.splitext(title)[0].replace(prefix, "")
        ax.set_title(title)
        if hide_axes:
            ax.set_axis_off()
    fig.colorbar(axim, ax=ax)
    plt.tight_layout()

    slclist = sario.parse_slclist_strings([os.path.split(s)[1] for s in filenames])
    plot_variances(slclist, imgs)
    return fig, axes, imgs


def plot_variances(slclist, imgs, title="Variance of SAR images"):
    fig, ax = plt.subplots()
    variances = np.var(imgs, axis=(1, 2))
    ax.plot(slclist, variances, "-x")
    ax.set_title(title)
    ax.set_ylabel("Variance of image")
    return fig, ax, variances


def _plot_rows_cols(num_imgs):
    nrow = int(np.floor(np.sqrt(num_imgs)))
    ncol = int(np.ceil(num_imgs / nrow))
    return nrow, ncol


#####################

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


# i## PART 2
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
        required=True,
        help="location of igram files. (default=%(default)s)",
    )
    p.add_argument(
        "--geo-dir",
        required=True,
        help="location of .geo SLC files. (default=%(default)s)",
    )
    p.add_argument(
        "--mag-thresh",
        default=3,
        type=float,
        help="Magnitude of earthquakes to threshold (default=%(default)s)",
    )
    p.add_argument(
        "--box-size",
        default=20,
        type=float,
        help="Size (in km) of box to subset around EQ hypocenter (default=%(default)s)",
    )
    p.add_argument(
        "--num-igrams",
        default=10,
        type=int,
        help="Desired number of independent igrams to include in stack (less used if not available, default=%(default)s)",
    )
    p.add_argument(
        "--ref-row",
        default=5,
        type=int,
        help="Row of reference point for stack calculation (default=%(default)s)",
    )
    p.add_argument(
        "--ref-col",
        default=5,
        type=int,
        help="Column of reference point for stack calculation (default=%(default)s)",
    )
    p.add_argument(
        "--eq-fname",
        default=TEXNET_DATA,
        help="filename of TexNet earthquake data (default=%(default)s)",
    )
    p.add_argument(
        "--igram-type",
        default="cross",
        choices=["cross", "pre", "post"],
        help=(
            "Type of igram selection for analyzing across, before, "
            "or after event (default=%(default)s)"
        ),
    )
    p.add_argument(
        "--no-outlier-removal",
        action="store_false",
        help="When set, skips averaging igram per date to discard bad dates (default=%(default)s)",
    )
    p.add_argument(
        "--no-ignore-geos",
        action="store_false",
        help="Skip applying the slclist_ignore.txt file (default=%(default)s)",
    )
    p.add_argument(
        "--no-plot",
        action="store_false",
        help="Skip plotting plots of results (default=%(default)s)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra debug info (default=%(default)s)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = get_cli_args()
    run_vrt_subset(
        args.igram_dir,
        args.geo_dir,
        mag_thresh=args.mag_thresh,
        eq_fname=args.eq_fname,
        box_size=args.box_size,
        remove_outliers=args.no_outlier_removal,
        ignore_geos=args.no_ignore_geos,
        num_igrams=args.num_igrams,
        ref=(args.ref_row, args.ref_col),
        igram_type=args.igram_type,
        verbose=args.verbose,
        plot_block=args.no_plot,
    )
