import datetime
import os
import numpy as np
import apertools.sario as sario
import coseismic_stack
import apertools.latlon as latlon
import apertools.subset as subset
import shapely.geometry
import rasterio as rio
import pandas as pd
import geopandas as gpd

from insar.prepare import remove_ramp
from insar.timeseries import PHASE_TO_CM

# TODO: merge overlapping date/areas... aftershocks means 3/26,27,29 all there

TEXNET_DATA = "/home/scott/cisr-data/texnet_events_20201012.csv"


def setup_folders(
    eq_fname=TEXNET_DATA,
    sar_fname="../S1A_20141215.geo.vrt",
    mag_thresh=3,
):
    df = load_eq_df(eq_fname=eq_fname, sar_fname=sar_fname, mag_thresh=mag_thresh)
    for (index, row) in df.iterrows():
        print(f"Creating folder {row.folder}")
        os.makedirs(row.folder, exist_ok=True)
    return df


def load_eq_df(
    eq_fname=TEXNET_DATA,
    sar_fname="../S1A_20141215.geo.vrt",
    mag_thresh=3,
    box_size=20,
):
    """Create a DataFrame of the top weekly earthquakes, limited to the
    area within sar_fname and above magnitude mag_thresh
    """
    eqdf = read_eqs(eq_fname)
    topeqdf = get_top_mag_dates(eqdf)
    df = get_eqs_in_bounds(sar_fname, topeqdf, mag_thresh=mag_thresh)
    df.index = pd.to_datetime(df.index)

    df["bbox"] = df.geometry.buffer(latlon.km_to_deg(box_size))
    df["folder"] = df.index.strftime("%Y%m%d").str.cat(df.event_id, sep="_")
    return df


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
    row has columns: max_mag,lat,lon,event_id,geometry,bbox,folder
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
        os.path.join(os.path.abspath(row.folder), f + ".vrt") for f in stack_fnames
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


def read_eqs(fname="../../fracking-qgis-data/texnet_events_20200726.csv"):
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
