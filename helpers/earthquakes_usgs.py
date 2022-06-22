import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
import requests

# TODO : ...is this easier in obspy?


"""
If I want to redo the query:

https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime=2014-11-11%2000:00:00&endtime=2020-11-18%2023:59:59&maxlatitude=50&minlatitude=24.6&maxlongitude=-65&minlongitude=-125&minmagnitude=2.5&orderby=time
"""
QUERY_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime={ymd1}"
    "%2000:00:00&endtime={ymd2}%2023:59:59&minmagnitude={mag}"
    # Boundaries are for CONUS, not alaska
    "&maxlatitude=50&minlatitude=24.6&maxlongitude=-65&minlongitude=-125"
)


def download_query(start_date, end_date, outname="query.csv", mag=2.5):
    """YYYY-mm-dd"""
    url = QUERY_URL.format(ymd1=start_date, ymd2=end_date, mag=mag)
    resp = requests.get(url)
    resp.raise_for_status()
    with open(outname, "w") as f:
        f.write(resp.text)


def load_query(filename):
    df = pd.read_csv(filename, infer_datetime_format=True, parse_dates=[0])
    df["year"] = df.time.dt.year
    # df["state"] = df.place.str.split(",").str[-1].str.strip()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    return merge_to_state(gdf)


def merge_to_state(gdf):
    states = gpd.read_file("state_boundaries/cb_2018_us_state_5m.shp")
    state_eq_df = sjoin(gdf, states, how="inner")
    state_eq_df.rename(columns={"NAME": "state"}, inplace=True)
    return state_eq_df


def get_eqs_by_state_year(filename):
    gdf = load_query(filename)
    cc = gdf.value_counts(["year", "state"])
    return (
        cc.reset_index()
        .rename({0: "count"}, axis="columns")
        .sort_values(by=["year", "count"], axis=0, ascending=False)
    )


# Getting counties within texas shape
# counties = gpd.read_file("countyl010g_shp_nt00964/countyl010g.shp")
# texas = states[states.NAME == 'Texas']
# texas_counties = sjoin(counties, texas, how='inner', op='within')
# gdf = gpd.GeoDataFrame(topdf, geometry=gpd.points_from_xy(topdf.lon, topdf.lat))

# download_query("2017-01-01", "2019-12-31", outname="query_17_19_m3.csv", mag="3.0")
# df_m3 = pd.concat([get_eqs_by_state_year(f) for f in sorted(glob.glob("query_*m3*csv"), reverse=True)]).reset_index()
# top7_states = df.head(7).state
# df[df.state.isin(top7_states)]