"""gps.py
Utilities for integrating GPS with InSAR maps

Links:

1. list of LLH for all gps stations: ftp://gneiss.nbmg.unr.edu/rapids/llh
Note: ^^ This file is stored in the `STATION_LLH_FILE`

2. Clickable names to explore: http://geodesy.unr.edu/NGLStationPages/GlobalStationList
3. Map of stations: http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html

"""
from __future__ import division, print_function
import os
import difflib  # For station name misspelling checks
import datetime
import requests
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.dates as mdates

from scipy.ndimage.filters import uniform_filter1d
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

import apertools
import apertools.utils
import apertools.sario
import apertools.plotting
from apertools.log import get_log

logger = get_log()

# URL for ascii file of 24-hour final GPS solutions in east-north-vertical (NA12)
GPS_BASE_URL = "http://geodesy.unr.edu/gps_timeseries/tenv3/NA12/{station}.NA12.tenv3"

# Assuming the master station list is in the
DIRNAME = os.path.dirname(os.path.abspath(__file__))
STATION_LLH_FILE = os.environ.get(
    'STATION_LIST',
    os.path.join(DIRNAME, 'data/station_llh_all.csv'),
)
START_DATE = datetime.date(2014, 11, 1)  # Start of InSAR data I care about


def _get_gps_dir():
    path = apertools.utils.get_cache_dir(force_posix=True)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


GPS_DIR = _get_gps_dir()


def plot_insar_vs_gps(geo_path=None,
                      defo_filename="deformation.npy",
                      station_name_list=None,
                      df=None,
                      kind="line",
                      reference_station=None,
                      **kwargs):
    """Make a GPS vs InSAR plot.

    kinds:
        line: plot out full data for each station
        errorbar: predict cumulative value for each station with error bars
        slope: plot gps value vs predicted insar (with 1-1 slope being perfect),
            gives insar error bars

    If reference_station is provided, all columns are centered to that series
        with gps subtracting the gps, insar subtracting the insar
    """

    igrams_dir = os.path.join(geo_path, 'igrams')
    if df is None:
        df = create_insar_gps_df(
            geo_path,
            defo_filename=defo_filename,
            # station_name_list=station_name_list,  # For now, filter given names after grabbing all
            reference_station=reference_station,
            **kwargs)
        # window_size=1, days_smooth_insar=5, days_smooth_gps=30):
    if station_name_list is not None:
        df = _filter_df_by_stations(df, station_name_list, igrams_dir, defo_filename)

    return plot_insar_gps_df(df, kind=kind, **kwargs)


def _filter_df_by_stations(df, station_name_list, igrams_dir, defo_filename):
    station_name_list = _load_station_list(
        igrams_dir=igrams_dir,
        defo_filename=defo_filename,
        station_name_list=station_name_list,
    )
    select_cols = [col for col in df.columns if any(name in col for name in station_name_list)]
    return df[select_cols]


def plot_insar_gps_df(df, kind="errorbar", grid=True, block=False, velocity=True, **kwargs):
    """Plot insar vs gps values from dataframe

    kinds:
        line: plot out full data for each station
        errorbar: predict cumulative value for each station with error bars
        slope: plot gps value vs predicted insar (with 1-1 slope being perfect),
            gives insar error bars
    """
    valid_kinds = ("line", "errorbar", "slope")

    # for idx, column in enumerate(columns):
    if kind == "errorbar":
        fig, axes = _plot_errorbar_df(df, velocity=velocity, **kwargs)
    elif kind == "line":
        fig, axes = _plot_line_df(df, **kwargs)
    elif kind == "slope":
        fig, axes = _plot_slope_df(df, **kwargs)
    else:
        raise ValueError("kind must be in: %s" % valid_kinds)
    fig.tight_layout()
    if grid:
        for ax in axes.ravel():
            ax.grid(True)

    plt.show(block=block)
    return fig, axes


def load_station_enu(station,
                     start_date=START_DATE,
                     end_date=None,
                     to_cm=True,
                     zero_mean=True,
                     zero_start=False):
    """Loads one gps station's ENU data since start_date until end_date
    as separate Series items

    Will scale to cm by default, and center the first data point
    to 0
    """
    station_df = load_station_data(station, to_cm=to_cm, start_date=start_date, end_date=end_date)

    if zero_start:
        start_val = station_df[['east', 'north', 'up']].iloc[:10].mean()
        enu_zeroed = station_df[['east', 'north', 'up']] - start_val
    elif zero_mean:
        mean_val = station_df[['east', 'north', 'up']].mean()
        enu_zeroed = station_df[['east', 'north', 'up']] - mean_val

    dts = station_df['dt']
    return dts, enu_zeroed


def load_station_data(station_name,
                      start_date=START_DATE,
                      end_date=None,
                      download_if_missing=True,
                      to_cm=True):
    """Loads one gps station's ENU data since start_date until end_date as a dataframe

    Args:
        station_name (str): 4 Letter name of GPS station
            See http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html for map
        start_date (datetime), default 2014-11-1, cutoff for beginning of GPS data
        end_date (datetime): default None, cut off for end of GPS data
        download_if_missing (bool): default True
    """
    station_name = station_name.upper()
    gps_data_file = os.path.join(GPS_DIR, '%s.NA12.tenv3' % station_name)
    if not os.path.exists(gps_data_file):
        logger.warning("%s does not exist.", gps_data_file)
        if download_if_missing:
            logger.info("Downloading %s", station_name)
            download_station_data(station_name)

    df = pd.read_csv(gps_data_file, header=0, sep=r"\s+")
    clean_df = _clean_gps_df(df, start_date, end_date)
    if to_cm:
        # logger.info("Converting %s GPS to cm" % station_name)
        clean_df[['east', 'north', 'up']] = 100 * clean_df[['east', 'north', 'up']]
    return clean_df


def _clean_gps_df(df, start_date=None, end_date=None):
    df['dt'] = pd.to_datetime(df['YYMMMDD'], format='%y%b%d')

    if start_date:
        df_ranged = df[df['dt'] >= start_date]
    if end_date:
        df_ranged = df_ranged[df_ranged['dt'] <= end_date]
    df_enu = df_ranged[['dt', '__east(m)', '_north(m)', '____up(m)']]
    df_enu = df_enu.rename(mapper=lambda s: s.replace('_', '').replace('(m)', ''), axis='columns')
    df_enu.reset_index(inplace=True, drop=True)
    return df_enu


def stations_within_image(image_ll=None, filename=None, mask_invalid=True, gps_filename=None):
    """Given an image, find gps stations contained in area

    Must have an associated .rsc file, either from being a LatlonImage (passed
    by image_ll), or have the file in same directory as `filename`

    Args:
        image_ll (LatlonImage): LatlonImage of area with data
        filename (str): filename to load into a LatlonImage
        mask_invalid (bool): Default true. if true, don't return stations
            where the image value is NaN or exactly 0

    Returns:
        ndarray: Nx3, with columns ['name', 'lon', 'lat']
    """
    if image_ll is None:
        image_ll = apertools.latlon.LatlonImage(filename=filename)

    df = read_station_llas(filename=gps_filename)
    station_lon_lat_arr = df[['lon', 'lat']].values
    contains_bools = image_ll.contains(station_lon_lat_arr)
    candidates = df[contains_bools][['name', 'lon', 'lat']].values
    good_stations = []
    if mask_invalid:
        for name, lon, lat in candidates:
            val = image_ll[..., lat, lon]
            if np.isnan(val):  # or val == 0: TODO: with window 1 reference, it's 0
                continue
            else:
                good_stations.append([name, lon, lat])
    else:
        good_stations = candidates
    return good_stations


def plot_stations(image_ll=None,
                  filename=None,
                  directory=None,
                  full_path=None,
                  mask_invalid=True,
                  station_name_list=None):
    """Plot an GPS station points contained within an image

    To only plot subset of stations, pass an iterable of strings to station_name_list
    """
    directory, filename, full_path = apertools.sario.get_full_path(directory, filename, full_path)

    if image_ll is None:
        try:
            # First check if we want to load the 3D stack "deformation.npy"
            image_ll = apertools.latlon.load_deformation_img(filename=filename, full_path=full_path)
        except ValueError:
            image_ll = apertools.latlon.LatlonImage(filename=filename)

    if mask_invalid:
        try:
            stack_mask = apertools.sario.load_mask(
                directory=directory, deformation_filename=full_path)
            image_ll[stack_mask] = np.nan
        except Exception:
            logger.warning("error in load_mask", exc_info=True)

    stations = stations_within_image(image_ll, mask_invalid=mask_invalid)
    if station_name_list:
        stations = [s for s in stations if s[0] in station_name_list]

    # TODO: maybe just plot_image_shifted
    fig, ax = plt.subplots()
    axim = ax.imshow(image_ll, extent=image_ll.extent)
    fig.colorbar(axim)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(stations)))
    # names, lons, lats = stations.T
    # ax.scatter(lons.astype(float), lats.astype(float), marker='X', label=names, c=color)
    size = 25
    for idx, (name, lon, lat) in enumerate(stations):
        # ax.plot(lon, lat, 'X', markersize=15, label=name)
        ax.scatter(lon, lat, marker='X', label=name, color=colors[idx], s=size)
    plt.legend()


def save_station_points_kml(station_iter):
    for name, lat, lon, alt in station_iter:
        apertools.kml.create_kml(
            title=name,
            desc='GPS station location',
            lon_lat=(lon, lat),
            kml_out='%s.kml' % name,
            shape='point',
        )


@lru_cache()
def read_station_llas(header=None, filename=None):
    """Read in the name, lat, lon, alt list of gps stations

    Assumes file is a csv with "name,lat,lon,alt" as columns
    Must give "header" argument if there is a header
    """
    filename = filename or STATION_LLH_FILE
    logger.debug("Searching %s for gps data" % filename)
    df = pd.read_csv(filename, header=header)
    df.columns = ['name', 'lat', 'lon', 'alt']
    return df


def station_lonlat(station_name):
    """Return the (lon, lat) of a `station_name`"""
    df = read_station_llas()
    station_name = station_name.upper()
    if station_name not in df['name'].values:
        closest_names = difflib.get_close_matches(station_name, df['name'], n=5)
        raise ValueError("No station named %s found. Closest: %s" % (station_name, closest_names))
    name, lat, lon, alt = df[df['name'] == station_name].iloc[0]
    return lon, lat


def station_rowcol(station_name=None, rsc_data=None):
    lon, lat = station_lonlat(station_name)
    return apertools.latlon.latlon_to_rowcol(lat, lon, rsc_data)


def station_distance(station_name1, station_name2):
    lonlat1 = station_lonlat(station_name1)
    lonlat2 = station_lonlat(station_name2)
    return apertools.latlon.latlon_to_dist(lonlat1[::-1], lonlat2[::-1])


def stations_within_rsc(rsc_filename=None, rsc_data=None, gps_filename=None):
    if rsc_data is None:
        if rsc_filename is None:
            raise ValueError("Need rsc_data or rsc_filename")
        rsc_data = apertools.sario.load(rsc_filename)

    df = read_station_llas(filename=gps_filename)
    station_lon_lat_arr = df[['lon', 'lat']].values
    contains_bools = [apertools.latlon.grid_contains(s, **rsc_data) for s in station_lon_lat_arr]
    return df[contains_bools][['name', 'lon', 'lat']].values


def download_station_data(station_name):
    station_name = station_name.upper()
    url = GPS_BASE_URL.format(station=station_name)
    response = requests.get(url)

    stationfile = url.split("/")[-1]  # Just blah.NA12.tenv3
    filename = "{}/{}".format(GPS_DIR, stationfile)
    logger.info("Saving to {}".format(filename))

    with open(filename, "w") as f:
        f.write(response.text)


def station_std(station, to_cm=True, start_date=START_DATE, end_date=None):
    """Calculates the sum of east, north, and vertical stds of gps"""
    dts, enu_df = load_station_enu(station, start_date=start_date, end_date=end_date, to_cm=to_cm)
    if enu_df.empty:
        logger.warning("%s gps data returned an empty dataframe")
        return np.nan
    return np.sum(enu_df.std())


def plot_gps_enu(station=None, days_smooth=12, start_date=START_DATE, end_date=None):
    """Plot the east,north,up components of `station`"""

    def remove_xticks(ax):
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

    dts, enu_df = load_station_enu(station, start_date=start_date, end_date=end_date, to_cm=True)
    (east_cm, north_cm, up_cm) = enu_df[['east', 'north', 'up']].T.values

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(dts, east_cm, 'b.')
    axes[0].set_ylabel('east (cm)')
    # axes[0].plot(dts, moving_average(east_cm, days_smooth), 'r-')
    axes[0].plot(dts, pd.Series(east_cm).rolling(days_smooth, min_periods=10).mean(), 'r-')
    axes[0].grid(True)
    remove_xticks(axes[0])

    axes[1].plot(dts, north_cm, 'b.')
    axes[1].set_ylabel('north (cm)')
    axes[1].plot(dts, pd.Series(north_cm).rolling(days_smooth, min_periods=10).mean(), 'r-')
    axes[1].grid(True)
    remove_xticks(axes[1])

    axes[2].plot(dts, up_cm, 'b.')
    axes[2].set_ylabel('up (cm)')
    axes[2].plot(dts, pd.Series(up_cm).rolling(days_smooth, min_periods=10).mean(), 'r-')
    axes[2].grid(True)
    # remove_xticks(axes[2])

    fig.suptitle(station)

    return fig, axes


def load_gps_los_data(
        geo_path=None,
        los_map_file=None,
        station_name=None,
        to_cm=True,
        zero_mean=True,
        zero_start=False,
        start_date=START_DATE,
        end_date=None,
        reference_station=None,
        enu_coeffs=None,
):
    """Load the GPS timeseries of a station name

    Returns the timeseries, and the datetimes of the points

    This assumes that the los is AWAY from the satellite towards the ground.
    We multiply the GPS data by -1 here so that ground moving upward
    (towards the mostly-pointing-down satellite) shows positive value.
    """
    if enu_coeffs is None:
        lon, lat = station_lonlat(station_name)
        enu_coeffs = apertools.los.find_enu_coeffs(
            lon, lat, geo_path=geo_path, los_map_file=los_map_file)

    df = load_station_data(station_name, to_cm=to_cm, start_date=start_date, end_date=end_date)
    # See docstring about negative 1 so vertical up ground gives a positive value
    enu_data = -1 * df[['east', 'north', 'up']].T
    los_gps_data = apertools.los.project_enu_to_los(enu_data, enu_coeffs=enu_coeffs)

    if zero_start:
        logger.debug("Resetting GPS data start to 0")
        los_gps_data = los_gps_data - np.mean(los_gps_data[:100])
    elif zero_mean:
        logger.debug("Making GPS data 0 mean")
        los_gps_data = los_gps_data - np.mean(los_gps_data)

    if reference_station is not None:
        dt_ref, losref = load_gps_los_data(geo_path, los_map_file, reference_station, to_cm,
                                           zero_mean, zero_start, start_date, end_date)
        return _merge_los(df['dt'], los_gps_data, dt_ref, losref)

    return df['dt'], los_gps_data


def _merge_los(dt1, los1, dt_ref, los_ref):
    df1 = pd.DataFrame(data={"g1": los1, "dt": dt1})
    df_ref = pd.DataFrame(data={"gref": los_ref, "dt": dt_ref})

    start = np.min(pd.concat([dt1, dt_ref]))
    end = np.max(pd.concat([dt1, dt1]))
    dt_merged = pd.date_range(start=start, end=end)

    df = pd.DataFrame(data={"dt": dt_merged})
    df = pd.merge(df, df1, on="dt", how="left")
    df = pd.merge(df, df_ref, on="dt", how="left")
    df.dropna(inplace=True)
    return df["dt"], (df["g1"] - df["gref"]).values


def moving_average(arr, window_size=7):
    """Takes a 1D array and returns the running average of same size"""
    if not window_size:
        return arr
    return uniform_filter1d(arr, size=window_size, mode='nearest')


def find_insar_ts(defo_filename='deformation.h5', station_name_list=[], window_size=1):
    """Get the insar timeseries closest to a list of GPS stations

    Returns the timeseries, and the datetimes of points for plotting

    Args:
        defo_filename
        station_name_list
        window_size (int): number of pixels in sqaure to average for insar timeseries
    """
    # geolist, deformation_stack = apertools.sario.load_deformation(full_path=defo_filename)
    # defo_img = apertools.latlon.load_deformation_img(full_path=defo_filename)

    insar_ts_list = []
    for station_name in station_name_list:
        lon, lat = station_lonlat(station_name)
        # row, col = defo_img.nearest_pixel(lat=lat, lon=lon)
        dem_rsc = apertools.sario.load_dem_from_h5(defo_filename)
        row, col = apertools.latlon.nearest_pixel(dem_rsc, lon=lon, lat=lat)
        insar_ts_list.append(
            get_stack_timeseries(
                defo_filename, row, col, station=station_name, window_size=window_size))

    geolist = apertools.sario.load_geolist_from_h5(defo_filename)
    return geolist, insar_ts_list


def get_stack_timeseries(filename,
                         row,
                         col,
                         stack_dset_name=apertools.sario.STACK_DSET,
                         station=None,
                         window_size=1):
    with h5py.File(filename, 'a') as f:
        try:
            dset = f[station]
            logger.debug("Getting cached timeseries at %s" % station)
            return dset[:]
        except (TypeError, KeyError):
            pass

        dset = f[stack_dset_name]
        logger.debug("Reading timeseries at %s from %s" % (station, filename))
        ts = apertools.utils.window_stack(dset, row, col, window_size)
        if station is not None:
            f[station] = ts
        return ts


def _series_to_date(series):
    return pd.to_datetime(series).apply(lambda row: row.date())


def create_insar_gps_df(geo_path,
                        defo_filename='deformation.h5',
                        station_name_list=[],
                        reference_station=None,
                        window_size=1,
                        days_smooth_insar=None,
                        days_smooth_gps=None,
                        **kwargs):
    """Set days_smooth to None or 0 to avoid any data smoothing

    If refernce station is specified, all timeseries will subtract
    that station's data
    """
    if not station_name_list:
        igrams_dir = os.path.join(geo_path, 'igrams')
        station_name_list = _load_station_list(
            igrams_dir=igrams_dir,
            defo_filename=defo_filename,
            station_name_list=station_name_list,
        )

    insar_df = create_insar_df(defo_filename, station_name_list, window_size, days_smooth_insar)
    gps_df = create_gps_los_df(geo_path, station_name_list, days_smooth_gps)
    df = combine_insar_gps_dfs(insar_df, gps_df)
    if reference_station:
        df = subtract_reference(df, reference_station)
    return _remove_bad_cols(df)


def _find_bad_cols(df, nan_threshold):
    empty_starts = df.columns[df.head(10).isna().all()]
    empty_ends = df.columns[df.tail(10).isna().all()]
    nan_pcts = df.isna().sum() / len(df)
    high_pct_nan = df.columns[nan_pcts > nan_threshold]
    high_pct_nan = [c for c in high_pct_nan if 'gps' in c]  # Ignore all the insar nans

    return np.concatenate((
        np.array(empty_starts),
        np.array(empty_ends),
        np.array(high_pct_nan),
    ))


def _remove_bad_cols(df, nan_threshold=.4):
    """Drops columns that are all NaNs, or where GPS doesn't cover whole period"""
    bad_cols = _find_bad_cols(df, nan_threshold)
    logger.info("Removing the following bad columns:")
    logger.info(bad_cols)

    for col in bad_cols:
        if col not in df.columns:
            continue
        station = col.replace("_gps", "").replace("_insar", "")
        df.drop("%s_gps" % station, axis=1, inplace=True)
        df.drop("%s_insar" % station, axis=1, inplace=True)
    return df


def subtract_reference(df, reference_station):
    """Center all columns of `df` based on the `reference_station` columns"""
    gps_ref_col = "%s_%s" % (reference_station, "gps")
    insar_ref_col = "%s_%s" % (reference_station, "insar")
    df_out = df.copy()
    for col in df.columns:
        if 'gps' in col:
            df_out[col] = df[col] - df[gps_ref_col]
        elif 'insar' in col:
            df_out[col] = df[col] - df[insar_ref_col]
    return df_out


def create_insar_df(defo_filename='deformation.h5',
                    station_name_list=[],
                    window_size=1,
                    days_smooth=5):
    """Set days_smooth to None or 0 to avoid any data smoothing"""
    geolist, insar_ts_list = find_insar_ts(
        defo_filename=defo_filename,
        station_name_list=station_name_list,
        window_size=window_size,
    )
    insar_df = pd.DataFrame({'dts': _series_to_date(pd.Series(geolist))})
    for stat, ts in zip(station_name_list, insar_ts_list):
        insar_df[stat + "_insar"] = moving_average(ts, days_smooth)
        # insar_df[stat + "_smooth_insar"] = moving_average(ts, days_smooth)
    return insar_df


def create_gps_los_df(geo_path, station_name_list=[], days_smooth=30):
    df_list = []
    for stat in station_name_list:
        gps_dts, los_gps_data = load_gps_los_data(geo_path, stat)

        df = pd.DataFrame({"dts": _series_to_date(gps_dts)})
        df[stat + "_gps"] = moving_average(los_gps_data, days_smooth)
        # df[stat + "_smooth_gps"] = moving_average(los_gps_data, days_smooth)
        df_list.append(df)

    min_date = np.min(pd.concat([df['dts'] for df in df_list]))
    max_date = np.max(pd.concat([df['dts'] for df in df_list]))

    # Now merge together based on uneven time data
    gps_df = pd.DataFrame({"dts": pd.date_range(start=min_date, end=max_date).date})
    for df in df_list:
        gps_df = pd.merge(gps_df, df, on="dts", how="left")

    return gps_df


def combine_insar_gps_dfs(insar_df, gps_df):
    # First constrain the date range to just the InSAR min/max dates
    df = pd.DataFrame({
        "dts":
        pd.date_range(start=np.min(insar_df["dts"]), end=np.max(insar_df["dts"])).date
    })
    df = pd.merge(df, insar_df, on="dts", how="left")
    df = pd.merge(df, gps_df, on="dts", how="left", suffixes=("", "_gps"))
    # Now final df has datetime as index
    return df.set_index("dts")


def create_gps_enu_df(station_name_list=None,
                      defo_filename=None,
                      igrams_dir=None,
                      start_date=START_DATE,
                      end_date=None,
                      days_smooth=None,
                      to_cm=True):
    if not station_name_list:
        station_name_list = _load_station_list(
            igrams_dir=igrams_dir,
            defo_filename=defo_filename,
            station_name_list=station_name_list,
        )

    df_list = []
    for station in station_name_list:
        dts, enu_df = load_station_enu(
            station, start_date=start_date, end_date=end_date, to_cm=to_cm)

        enu_df["dts"] = _series_to_date(dts)
        # Add station name to columns for post-merge identification
        mapper = {
            direction: "{}_{}".format(station, direction)
            for direction in ("east", "north", "up")
        }
        enu_df.rename(mapper, inplace=True, axis=1)
        df_list.append(enu_df)

    min_date = np.min(pd.concat([df['dts'] for df in df_list]))
    max_date = np.max(pd.concat([df['dts'] for df in df_list]))

    # Now merge together based on uneven time data
    enu_df = pd.DataFrame({"dts": pd.date_range(start=min_date, end=max_date).date})
    for df in df_list:
        enu_df = pd.merge(enu_df, df, on="dts", how="left")

    enu_df = enu_df.set_index("dts")
    nan_thresh = .8
    # Drop empty columns or no data for last 14 days
    bad_cols = _find_bad_cols(enu_df, nan_thresh)
    for col in np.unique(bad_cols):
        logger.info("Dropping %s", col)
        enu_df.drop(col, axis=1, inplace=True)

    return enu_df


def get_final_east_values(east_df):
    stations, vals = [], []

    direc = None
    for column, val in east_df.tail(14).mean().items():
        station, d = column.split("_")
        direc = d
        stations.append(station)
        vals.append(val)
    return pd.DataFrame(index=stations, data={direc: vals})


def fit_date_series(series):
    """Fit a line to `series` with (possibly) uneven dates as index.

    Can be used to detrend, or predict final value

    Returns:
        Series: a line, equal length to arr, with same index as `series`
    """
    # TODO: check that subtracting first item doesn't change it
    full_dates = series.index  # Keep for final series formation
    full_date_idxs = mdates.date2num(full_dates)

    series_clean = series.dropna()
    idxs = mdates.date2num(series_clean.index)

    coeffs = np.polyfit(idxs, series_clean, 1)
    poly = np.poly1d(coeffs)

    line_fit = poly(full_date_idxs)
    return pd.Series(line_fit, index=full_dates)


def flat_std(series):
    """Find the std dev of an Series with a linear component removed"""
    return np.std(series - fit_date_series(series))


def _plot_errorbar_df(df, ylim=None, velocity=True, **kwargs):
    gps_cols, insar_cols, final_gps_vals, final_insar_vals = get_final_gps_insar_values(
        df, velocity=velocity, **kwargs)
    gps_stds = [flat_std(df[col].dropna()) for col in df.columns if col in gps_cols]

    fig, axes = plt.subplots(squeeze=False)
    ax = axes[0, 0]
    idxs = range(len(final_gps_vals))
    ax.errorbar(idxs, final_gps_vals, gps_stds, marker='o', lw=2, linestyle='', capsize=6)
    ax.plot(idxs, final_insar_vals, 'rx')

    if velocity:
        ax.set_ylabel('mm/year of LOS displacement')
    else:
        ax.set_ylabel('CM of cumulative LOS displacement')

    labels = [c.replace('_gps', '') for c in gps_cols]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation='vertical', fontsize=12)
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, axes


def _plot_slope_df(df, **kwargs):
    gps_cols, insar_cols, final_gps_vals, final_insar_vals = get_final_gps_insar_values(df)
    insar_stds = [flat_std(df[col].dropna()) for col in df.columns if col in insar_cols]

    fig, axes = plt.subplots(squeeze=False)
    ax = axes[0, 0]
    ax.errorbar(final_gps_vals, final_insar_vals, yerr=insar_stds, fmt='rx', capsize=6)

    max_val = max(np.max(final_insar_vals), np.max(final_gps_vals))
    min_val = min(np.min(final_insar_vals), np.min(final_gps_vals))
    ax.plot(np.linspace(min_val, max_val), np.linspace(min_val, max_val), 'g')
    ax.set_ylabel('InSAR predicted cumulative CM')
    ax.set_xlabel('GPS cumulative CM')
    return fig, axes


def _plot_line_df(df, ylim=None, share=True, days_smooth_gps=None, days_smooth_insar=None,
                  **kwargs):
    """share is used to indicate that GPS and insar will be on same axes"""

    def _plot_smoothed(ax, df, column, days_smooth, marker):
        ax.plot(
            df[column].dropna().index,
            df[column].dropna().rolling(days_smooth_gps).mean(),
            marker,
            linewidth=3,
            label="%s day smoothed %s" % (days_smooth, column))

    columns = df.columns
    nrows = 1 if share else 2
    fig, axes = plt.subplots(nrows, len(columns) // 2, figsize=(16, 5), squeeze=False)

    gps_idxs = np.where(['gps' in col for col in columns])[0]
    insar_idxs = np.where(['insar' in col for col in columns])[0]

    for idx, column in enumerate(columns):
        if 'insar' in column:
            marker = 'rx'
            ax_idx = np.where(insar_idxs == idx)[0][0] if share else idx
        else:
            marker = 'b.'
            ax_idx = np.where(gps_idxs == idx)[0][0] if share else idx

        ax = axes.ravel()[ax_idx]
        # axes[idx].plot(df.index, df[column].fillna(method='ffill'), marker)
        ax.plot(df.index, df[column], marker, label=column)

        ax.set_title(column)
        if ylim is not None:
            ax.set_ylim(ylim)
        xticks = ax.get_xticks()
        ax.set_xticks([xticks[0], xticks[len(xticks) // 2], xticks[-1]])
        if days_smooth_gps and 'gps' in column:
            _plot_smoothed(ax, df, column, days_smooth_gps, "b-")
        if days_smooth_insar and 'insar' in column:
            _plot_smoothed(ax, df, column, days_smooth_insar, "r-")

        ax.set_ylabel('InSAR cumulative CM')
        ax.legend()

    axes.ravel()[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    return fig, axes


def _get_gps_insar_cols(df):
    gps_idxs = ['gps' in col for col in df.columns]
    insar_idxs = ['insar' in col for col in df.columns]
    gps_cols = df.columns[gps_idxs]
    insar_cols = df.columns[insar_idxs]
    return gps_idxs, gps_cols, insar_idxs, insar_cols


def _fit_line_to_dates(df):
    return np.array([fit_date_series(df[col]).tail(1).squeeze() for col in df.columns])


def get_final_gps_insar_values(df, linear=True, as_df=False, velocity=True):
    # TODO: get rid of the "linear fit to unregularized"
    if linear:
        final_val_arr = _fit_line_to_dates(df)
    else:
        final_val_arr = df.tail(10).mean().values
    if velocity:
        full_dates = df.index
        days_spanning = (full_dates[-1] - full_dates[0]).days
        final_val_arr *= (10 * 365 / days_spanning)  # Now in mm/year

    gps_idxs, gps_cols, insar_idxs, insar_cols = _get_gps_insar_cols(df)

    final_gps_vals = final_val_arr[gps_idxs]
    final_insar_vals = final_val_arr[insar_idxs]
    if not as_df:
        return gps_cols, insar_cols, final_gps_vals, final_insar_vals
    else:
        final_val_station_order = [s.split('_')[0] for s in gps_cols]
        return pd.DataFrame(
            index=final_val_station_order, data={
                'gps': final_gps_vals,
                'insar': final_insar_vals
            })


def _load_stations(igrams_dir=None, defo_filename=None, defo_full_path=None):
    directory, filename, full_path = apertools.sario.get_full_path(igrams_dir, defo_filename,
                                                                   defo_full_path)
    defo_img = apertools.latlon.load_deformation_img(filename=filename, full_path=full_path)
    existing_station_tuples = stations_within_image(defo_img)
    return existing_station_tuples


def _load_station_list(igrams_dir=None,
                       defo_filename=None,
                       defo_full_path=None,
                       station_name_list=[]):
    existing_station_tuples = _load_stations(igrams_dir, defo_filename, defo_full_path)
    if not station_name_list:
        # Take all station names
        station_name_list = [name for name, lon, lat in existing_station_tuples]
    else:
        station_name_list = [
            name for name, lon, lat in existing_station_tuples if name in station_name_list
        ]
    return sorted(station_name_list, key=lambda name: station_name_list.index(name))


def _stations_from_df(df):
    """Takes df with columns ['NMHB_insar', 'TXAD_insar',...], returns unique station names"""
    return list(set(map(lambda s: s.split('_')[0], df.columns)))


def create_station_location_df(timeseries_df=None, station_name_list=None):
    """Start with a list of station names or a df
    with index of `dts` and columns as ['NMHB_gps', 'TXAD_insar'...],
    Create a dataframe like

    station_name |     lon    |   lat
    ------------------------------------
     TXKM        |  -103.108  | 31.8426

    Index is station_name, so lon/lat can be accessed with `df.loc['TXKM']`
    """
    if station_name_list is None:
        station_name_list = _stations_from_df(timeseries_df)
    station_lonlats = [station_lonlat(name) for name in station_name_list]
    lons, lats = zip(*station_lonlats)
    return pd.DataFrame(index=station_name_list, data={'lon': lons, 'lat': lats})


def _get_residuals(df, which):
    if which not in ('gps', 'insar', 'diff'):
        raise ValueError("argument `which` must in ('diff','gps','insar')")

    df_final_vals = get_final_gps_insar_values(df, as_df=True)
    df_locations = create_station_location_df(df)
    df_merged = df_locations.join(df_final_vals)

    if which == 'diff':
        values = df_merged['gps'] - df_merged['insar']
    else:
        values = df_merged[which]
    return df_merged, values


def plot_residuals_by_loc(df,
                          which='diff',
                          title=None,
                          fig=None,
                          ax=None,
                          plot_scatter=True,
                          **plot_kwargs):
    """Takes a timeseries df and plots the final values at their lat/lons

    df should be the timeseries df with "dts" as index created by create_insar_gps_df
    `which` argument is "diff","gps","insar", where 'diff' takes (gps - insar)
    """

    def _build_labels(df_merged, values):
        """Inclue station name and final value in label"""
        return ["{}:\n{:.2f}".format(name, val) for name, val in zip(df_merged.index, values)]

    df_merged, values = _get_residuals(df, which)

    print("Total abs values summed:", total_abs_error(values))
    print("RMS value:", rms(values))

    xs = df_merged['lon']
    ys = df_merged['lat']
    labels = _build_labels(df_merged, values)
    if title is None:
        title = "Final values of %s by location" % which
    fig, ax = _plot_latlon_with_labels(
        xs, ys, values, labels, title="", fig=fig, ax=ax, plot_scatter=plot_scatter, **plot_kwargs)
    return df_merged, fig, ax


def rms(values):
    return np.sqrt(np.mean(np.square(values)))


def total_abs_error(errors):
    return np.sum(np.abs(errors))


def _plot_latlon_with_labels(xs,
                             ys,
                             values,
                             labels,
                             title="",
                             fig=None,
                             ax=None,
                             plot_scatter=True,
                             **plot_kwargs):
    fig, ax = apertools.plotting.get_fig_ax(fig, ax)

    if plot_scatter:
        axim = ax.scatter(xs, ys, c=values, zorder=10, **plot_kwargs)
        fig.colorbar(axim)

    for label, x, y in zip(labels, xs, ys):

        ax.annotate(
            label,
            xy=(x, y),
            xytext=(-10, 10),
            textcoords='offset points',
            # ha='right',
            # va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        )
    ax.set_title(title)
    return fig, ax


def plot_gps_east_by_loc(defo_filename,
                         igrams_dir,
                         start_date=START_DATE,
                         end_date=None,
                         cmap_name="seismic_r",
                         **plot_kwargs):
    enu_df = create_gps_enu_df(
        defo_filename=defo_filename,
        igrams_dir=igrams_dir,
        end_date=end_date,
        start_date=start_date,
    )
    east_cols = [col for col in enu_df if "east" in col]
    east_df = enu_df[east_cols]

    df_locations = create_station_location_df(east_df)
    df_final_vals = get_final_east_values(east_df)
    df_merged = df_locations.join(df_final_vals)

    labels = [
        "{}: {:.2f}".format(stat, val) for stat, val in zip(df_merged.index, df_merged['east'])
    ]
    xs = df_merged['lon']
    ys = df_merged['lat']
    vals = df_merged['east']
    # Note: reversed for ascending path so that red = west movement = toward satellite
    cmap = apertools.plotting.make_shifted_cmap(
        cmap_name=cmap_name, vmax=vals.max(), vmin=vals.min())

    first_date, last_date = east_df.index[0], east_df.index[-1]
    title = "east GPS movement from {} to {}".format(first_date, last_date)
    fig, axes = _plot_latlon_with_labels(
        xs, ys, vals, labels, title=title, cmap=cmap, **plot_kwargs)
    return df_merged, fig, axes


def get_mean_correlations(igrams_dir=None,
                          defo_filename=None,
                          defo_full_path=None,
                          cc_filename="cc_stack.h5"):
    existing_station_tuples = _load_stations(igrams_dir, defo_filename, defo_full_path)
    corrs = {}
    dem_rsc = apertools.sario.load_dem_from_h5(defo_filename)
    with h5py.File(cc_filename) as f:
        for name, lon, lat in existing_station_tuples:
            row, col = apertools.latlon.latlon_to_rowcol(lat, lon, dem_rsc)
            corrs[name] = f["mean_stack"][row, col]
    return corrs
