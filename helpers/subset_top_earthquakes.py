from datetime import datetime
import os

import apertools.sario as sario
import coseismic_stack
import apertools.latlon as latlon
import apertools.subset as subset
import shapely.geometry
import rasterio as rio
import pandas as pd
import geopandas as gpd

# from insar.prepare import remove_ramp
# from insar.timeseries import PHASE_TO_CM


def read_eqs(fname="../../fracking-qgis-data/texnet_events_20200726.csv"):
    df = pd.read_csv(
        fname,
        usecols=(
            "EventID",
            "Origin Date",
            "Origin Time",
            "Latitude",
            "Longitude",
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
    dailydf = dailydf.groupby("date")["mag"].agg(max_mag="max", idxmax="idxmax")

    dailydf[["lat", "lon", "event_id"]] = df.loc[
        dailydf["idxmax"], ["lat", "lon", "event_id"]
    ].values
    dailydf = (
        dailydf[["max_mag", "lat", "lon", "event_id"]]
        .sort_values("max_mag", ascending=False)
        .head(n)
    )
    return gpd.GeoDataFrame(
        dailydf, geometry=gpd.points_from_xy(dailydf.lon, dailydf.lat)
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


def setup_folders(
    eq_fname="/home/scott/cisr-data/texnet_events_20200726.csv",
    sar_fname="../S1A_20141215.geo.vrt",
    mag_thresh=3,
):
    eqdf = read_eqs(eq_fname)
    topeqdf = get_top_mag_dates(eqdf)
    df = get_eqs_in_bounds(sar_fname, topeqdf, mag_thresh=mag_thresh)
    df.index = pd.to_datetime(df.index)

    df["bbox"] = df.geometry.buffer(latlon.km_to_deg(20))
    df["folder"] = df.index.strftime("%Y%m%d").str.cat(df.event_id, sep="_")
    for (index, row) in df.iterrows():
        print(f"Creating folder {row.folder}")
        os.makedirs(row.folder, exist_ok=True)
    return df


def subset_unws(
    event_date, row, igram_dir, geo_dir, ignore_geos=True, num_igrams=10, verbose=True
):
    """
    event_date is from index of df
    row has columns: max_mag,lat,lon,event_id,geometry,bbox,folder
    """
    gi_file = "geolist_ignore.txt" if ignore_geos else None
    geolist, intlist = sario.load_geolist_intlist(
        igram_dir, geo_dir=geo_dir, geolist_ignore_file=gi_file
    )
    stack_igrams = coseismic_stack.select_igrams(
        geolist, intlist, event_date, num_igrams=num_igrams
    )
    # stack_igrams = coseismic_stack.select_pre_event(geolist, intlist, event_date)
    # stack_igrams = select_post_event(geolist, intlist, event_date)

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
