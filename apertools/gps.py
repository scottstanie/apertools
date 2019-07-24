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
START_YEAR = 2014  # So far I don't care about older data


def _get_gps_dir():
    path = apertools.utils.get_cache_dir(force_posix=True)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


GPS_DIR = _get_gps_dir()


def load_station_enu(station, start_year=START_YEAR, end_year=None, to_cm=True):
    """Loads one gps station's ENU data since start_year until end_year
    as separate Series items

    Will scale to cm by default, and center the first data point
    to 0
    """
    station_df = load_station_data(station, to_cm=to_cm)

    enu_zeroed = station_df[['east', 'north', 'up']] - station_df[['east', 'north', 'up']].iloc[0]
    dts = station_df['dt']
    return dts, enu_zeroed


def load_station_data(station_name,
                      start_year=START_YEAR,
                      end_year=None,
                      download_if_missing=True,
                      to_cm=True):
    """Loads one gps station's ENU data since start_year until end_year as a dataframe

    Args:
        station_name (str): 4 Letter name of GPS station
            See http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html for map
        start_year (int), default 2014, cutoff for beginning of GPS data
        end_year (int): default None, cut off for end of GPS data
        download_if_missing (bool): default True
    """
    gps_data_file = os.path.join(GPS_DIR, '%s.NA12.tenv3' % station_name)
    if not os.path.exists(gps_data_file):
        logger.warning("%s does not exist.", gps_data_file)
        if download_if_missing:
            logger.info("Downloading %s", station_name)
            download_station_data(station_name)

    df = pd.read_csv(gps_data_file, header=0, sep=r"\s+")
    clean_df = _clean_gps_df(df, start_year, end_year)
    if to_cm:
        logger.info("Converting %s GPS to cm" % station_name)
        clean_df[['east', 'north', 'up']] = 100 * clean_df[['east', 'north', 'up']]
    return clean_df


def _clean_gps_df(df, start_year, end_year=None):
    df['dt'] = pd.to_datetime(df['YYMMMDD'], format='%y%b%d')

    df_ranged = df[df['dt'] >= datetime.datetime(start_year, 1, 1)]
    if end_year:
        df_ranged = df_ranged[df_ranged['dt'] <= datetime.datetime(end_year, 1, 1)]
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
            if np.isnan(val) or val == 0:
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
            stack_mask = apertools.sario.load_mask(directory=directory,
                                                   deformation_filename=full_path)
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
    logger.info("Searching %s for gps data" % filename)
    df = pd.read_csv(filename, header=header)
    df.columns = ['name', 'lat', 'lon', 'alt']
    return df


def station_lonlat(station_name):
    """Return the (lon, lat) of a `station_name`"""
    df = read_station_llas()
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
    url = GPS_BASE_URL.format(station=station_name)
    response = requests.get(url)

    stationfile = url.split("/")[-1]  # Just blah.NA12.tenv3
    filename = "{}/{}".format(GPS_DIR, stationfile)
    logger.info("Saving to {}".format(filename))

    with open(filename, "w") as f:
        f.write(response.text)


def station_std(station, to_cm=True, start_year=START_YEAR, end_year=None):
    """Calculates the sum of east, north, and vertical stds of gps"""
    dts, enu_df = load_station_enu(station, start_year=start_year, end_year=end_year, to_cm=to_cm)
    if enu_df.empty:
        logger.warning("%s gps data returned an empty dataframe")
        return np.nan
    return np.sum(enu_df.std())


# Alias for when we don't really care about the sqrt
station_variance = station_std


def plot_gps_enu(station=None, days_smooth=12, start_year=START_YEAR, end_year=None):
    """Plot the east,north,up components of `station`"""

    def remove_xticks(ax):
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

    dts, enu_df = load_station_enu(station, start_year=start_year, end_year=end_year, to_cm=True)
    (east_cm, north_cm, up_cm) = enu_df[['east', 'north', 'up']].T.values

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(dts, east_cm, 'b.')
    axes[0].set_ylabel('east (cm)')
    # axes[0].plot(dts, moving_average(east_cm, days_smooth), 'r-')
    axes[0].plot(dts, pd.Series(east_cm).rolling(days_smooth, min_periods=1).mean(), 'r-')
    axes[0].grid(True)
    remove_xticks(axes[0])

    axes[1].plot(dts, north_cm, 'b.')
    axes[1].set_ylabel('north (cm)')
    axes[1].plot(dts, pd.Series(north_cm).rolling(days_smooth, min_periods=1).mean(), 'r-')
    axes[1].grid(True)
    remove_xticks(axes[1])

    axes[2].plot(dts, up_cm, 'b.')
    axes[2].set_ylabel('up (cm)')
    axes[2].plot(dts, pd.Series(up_cm).rolling(days_smooth, min_periods=1).mean(), 'r-')
    axes[2].grid(True)
    # remove_xticks(axes[2])

    fig.suptitle(station)

    return fig, axes


def load_gps_los_data(
        geo_path,
        station_name=None,
        to_cm=True,
        zero_start=True,
        start_year=START_YEAR,
        end_year=None,
):
    """Load the GPS timeseries of a station name

    Returns the timeseries, and the datetimes of the points
    """
    lon, lat = station_lonlat(station_name)
    enu_coeffs = apertools.los.find_enu_coeffs(lon, lat, geo_path=geo_path)

    df = load_station_data(station_name, to_cm=to_cm)
    enu_data = df[['east', 'north', 'up']].T
    los_gps_data = apertools.los.project_enu_to_los(enu_data, enu_coeffs=enu_coeffs)

    if zero_start:
        logger.info("Resetting GPS data start to 0")
        los_gps_data = los_gps_data - np.mean(los_gps_data[:100])
    return df['dt'], los_gps_data


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
            get_stack_timeseries(defo_filename,
                                 row,
                                 col,
                                 station=station_name,
                                 window_size=window_size))

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
            logger.info("Getting cached timeseries at %s" % station)
            return dset[:]
        except (TypeError, KeyError):
            pass

        dset = f[stack_dset_name]
        logger.info("Reading timeseries at %s from %s" % (station, filename))
        ts = apertools.utils.window_stack(dset, row, col, window_size)
        if station is not None:
            f[station] = ts
        return ts


def _series_to_date(series):
    return pd.to_datetime(series).apply(lambda row: row.date())


def create_insar_gps_df(geo_path,
                        defo_filename='deformation.h5',
                        station_name_list=[],
                        window_size=1,
                        insar_days_smooth=5,
                        gps_days_smooth=30):
    """Set days_smooth to None or 0 to avoid any data smoothing"""
    insar_df = create_insar_df(defo_filename, station_name_list, window_size, insar_days_smooth)
    gps_df = create_gps_los_df(geo_path, station_name_list, gps_days_smooth)
    return combine_gps_insar_dfs(insar_df, gps_df)


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


def combine_gps_insar_dfs(insar_df, gps_df):
    # First constrain the date range to just the InSAR min/max dates
    df = pd.DataFrame(
        {"dts": pd.date_range(start=np.min(insar_df["dts"]), end=np.max(insar_df["dts"])).date})
    df = pd.merge(df, insar_df, on="dts", how="left")
    df = pd.merge(df, gps_df, on="dts", how="left", suffixes=("", "_gps"))
    # Now final df has datetime as index
    return df.set_index("dts")


def fit_date_series(series):
    """Fit a line to `series` with (possibly) uneven dates as index.

    Can be used to detrend, or predict final value

    Returns:
        Series: a line, equal length to arr, with same index as `series`
    """
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


def plot_insar_gps_df(df, kind="errorbar", grid=True, **kwargs):
    valid_kinds = ("line", "errbar", "slope")

    # for idx, column in enumerate(columns):
    if kind == "line":
        fig, axes = _plot_line_df(df, **kwargs)
    elif kind == "errorbar":
        fig, axes = _plot_errorbar_df(df, **kwargs)
    elif kind == "slope":
        fig, axes = _plot_slope_df(df, **kwargs)
    else:
        raise ValueError("kind must be in: %s" % valid_kinds)
    fig.tight_layout()
    if grid:
        for ax in axes.ravel():
            ax.grid(True)
    return fig, axes


def _final_vals(df, linear=True):
    if linear:
        final_vals = np.array([fit_date_series(df[col]).tail(1).squeeze() for col in df.columns])
    else:
        final_vals = df.tail(1).squeeze().values

    gps_idxs = ['gps' in col for col in df.columns]
    insar_idxs = ['insar' in col for col in df.columns]
    gps_cols = df.columns[gps_idxs]
    insar_cols = df.columns[insar_idxs]

    final_gps_vals = final_vals[gps_idxs]
    final_insar_vals = final_vals[insar_idxs]
    return gps_cols, insar_cols, final_gps_vals, final_insar_vals


def _plot_errorbar_df(df, ylims=None, **kwargs):
    # In [6]: df.std()
    # Out[6]:
    # TXKM_insar    0.053233
    # TXMH_insar    0.432821
    gps_cols, insar_cols, final_gps_vals, final_insar_vals = _final_vals(df, **kwargs)
    idxs = range(len(final_gps_vals))
    gps_stds = [flat_std(df[col].dropna()) for col in df.columns if col in gps_cols]

    fig, axes = plt.subplots(squeeze=False)
    ax = axes[0, 0]
    ax.errorbar(idxs, final_gps_vals, gps_stds, marker='o', lw=2, linestyle='', capsize=6)
    ax.plot(idxs, final_insar_vals, 'rx')

    labels = [c.strip('_gps') for c in gps_cols]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation='vertical', fontsize=12)
    if ylims is not None:
        ax.set_ylim(ylims)
    return fig, axes


def _plot_slope_df(df, **kwargs):
    gps_cols, insar_cols, final_gps_vals, final_insar_vals = _final_vals(df)
    insar_stds = [val for col, val in df.std().items() if col in insar_cols]

    fig, axes = plt.subplots(squeeze=False)
    ax = axes[0, 0]
    ax.errorbar(final_gps_vals, final_insar_vals, yerr=insar_stds, fmt='rx')

    max_val = max(np.max(final_insar_vals), np.max(final_gps_vals))
    min_val = min(np.min(final_insar_vals), np.min(final_gps_vals))
    ax.plot(np.linspace(min_val, max_val), np.linspace(min_val, max_val), 'g')
    return fig, axes


def _plot_line_df(df, fig, axes, ylims=None, **kwargs):
    columns = df.columns
    fig, axes = plt.subplots(2, len(columns) // 2, figsize=(16, 5), squeeze=False)

    for idx, column in enumerate(columns):
        ax = axes.ravel()[idx]
        if 'insar' in column:
            marker = 'r.'
        else:
            marker = 'b.'
        # axes[idx].plot(df.index, df[column].fillna(method='ffill'), marker)
        ax.plot(df.index, df[column], marker)
        ax.set_title(column)
        if ylims is not None:
            ax.set_ylim(ylims)
        xticks = ax.get_xticks()
        ax.set_xticks([xticks[0], xticks[len(xticks) // 2], xticks[-1]])
        # ax.locator_params(axis='x', tight=True, nbins=3)

    axes.ravel()[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    return fig, axes


def _load_station_list(igrams_dir, defo_filename, station_name_list):
    defo_img = apertools.latlon.load_deformation_img(igrams_dir, filename=defo_filename)
    existing_station_tuples = stations_within_image(defo_img)
    if station_name_list is not None:
        station_name_list = [
            name for name, lon, lat in existing_station_tuples if name in station_name_list
        ]
    return sorted(station_name_list, key=lambda name: station_name_list.index(name))


def plot_gps_vs_insar(geo_path, defo_filename="deformation.npy", station_name_list=None):
    """Make a single plot of GPS vs InSAR
    Note that without the `align` option, the two might differ by an angle
    due to differences in reference points for the InSAR
    """
    igrams_dir = os.path.join(geo_path, 'igrams')
    station_name_list = _load_station_list(igrams_dir, defo_filename, station_name_list)

    for station_name in station_name_list:
        plt.figure()

        gps_dts, los_gps_data = load_gps_los_data(geo_path, station_name)

        plt.plot(gps_dts, los_gps_data, 'b.', label='gps data: %s' % station_name)

        days_smooth = 60
        los_gps_data_smooth = moving_average(los_gps_data, days_smooth)
        plt.plot(gps_dts,
                 los_gps_data_smooth,
                 'b',
                 linewidth='4',
                 label='%d day smoothed gps data: %s' % (days_smooth, station_name))

        geolist, insar_ts_list = find_insar_ts(defo_filename=defo_filename,
                                               station_name_list=[station_name])
        insar_ts = insar_ts_list[0]

        plt.plot(geolist, insar_ts, 'rx', label='insar data', ms=5)

        days_smooth = 5
        insar_ts_smooth = moving_average(insar_ts, days_smooth)
        plt.plot(geolist,
                 insar_ts_smooth,
                 'r',
                 label='%s day smoothed insar' % days_smooth,
                 linewidth=3)

        plt.legend()
    plt.show()


def plot_gps_vs_insar_diff(geo_path,
                           fignum=None,
                           defo_filename='deformation.h5',
                           station_name_list=None):

    igrams_dir = os.path.join(geo_path, 'igrams')
    station_name_list = _load_station_list(igrams_dir, defo_filename, station_name_list)

    stat1, stat2 = station_name_list[:2]
    logger.info("Using stations: %s" % station_name_list[:2])

    # # First way: project each ENU to LOS, then subtract
    gps_dts1, los_gps_data1 = load_gps_los_data(geo_path, stat1, zero_start=False)
    gps_dts2, los_gps_data2 = load_gps_los_data(geo_path, stat2, zero_start=False)
    logger.info('first gps entries:')
    logger.info(los_gps_data1[0])
    logger.info(los_gps_data2[0])

    df1 = gps_dts1.to_frame()
    df1['gps1'] = los_gps_data1
    df2 = gps_dts2.to_frame()
    df2['gps2'] = los_gps_data2
    diff_df = pd.merge(df1, df2, how='inner', on=['dt'])
    # diff_df['d_gps'] = diff_df['gps1'] - diff_df['gps2']
    gps_diff_ts = diff_df['gps1'] - diff_df['gps2']
    gps_diff_ts = gps_diff_ts - np.mean(gps_diff_ts[:100])
    gps_dts = diff_df['dt']

    start_mean = np.mean(gps_diff_ts[:100])
    # start_mean = np.mean(los_gps_data2[:100])
    logger.info("Resetting GPS data start to 0 of station 2, subtract %f" % start_mean)
    gps_diff_ts = gps_diff_ts - start_mean

    plt.figure(fignum)
    plt.plot(gps_dts, gps_diff_ts, 'b.', label='gps data: %s' % stat1)

    days_smooth = 60
    los_gps_data_smooth = moving_average(gps_diff_ts, days_smooth)
    plt.plot(
        gps_dts,
        los_gps_data_smooth,
        'b',
        linewidth='4',
        # label='%d day smoothed gps data: %s' % (days_smooth, station_name))
        label='%d day smoothed gps data: %s-%s' % (days_smooth, stat1, stat2))

    geolist, (insar_ts1, insar_ts2) = find_insar_ts(defo_filename=defo_filename,
                                                    station_name_list=[stat1, stat2])

    insar_diff = insar_ts1 - insar_ts2

    plt.plot(geolist, insar_diff, 'r', label='insar data difference', ms=5)

    days_smooth = 5
    insar_diff_smooth = moving_average(insar_diff, days_smooth)
    insar_diff_smooth -= insar_diff_smooth[0]
    plt.plot(geolist,
             insar_diff_smooth,
             'r',
             label='linear %s day smoothed insar' % days_smooth,
             linewidth=3)

    # plt.ylim([-2, 2])
    plt.ylabel('cm of cumulative LOS displacement')
    plt.legend()

    return gps_dts, los_gps_data1, los_gps_data2, gps_diff_ts, geolist, insar_ts1, insar_ts2


def plot_all_gps_insar(geo_path,
                       fignum=None,
                       defo_filename='deformation.h5',
                       ref_station=None,
                       station_name_list=None):
    """
    For a subset, pass in the station_name_list
    """
    igrams_dir = os.path.join(geo_path, 'igrams')
    station_name_list = _load_station_list(igrams_dir, defo_filename, station_name_list)

    insar_df = create_insar_df(defo_filename=defo_filename, station_name_list=station_name_list)

    ref_dts, ref_los_gps_data = load_gps_los_data(geo_path,
                                                  station_name=ref_station,
                                                  zero_start=False)
    ref_df = ref_dts.to_frame()
    ref_df['ref_gps'] = ref_los_gps_data

    # First way: project each ENU to LOS, then subtract
    # gps_dt_list, los_gps_list = [], []
    df_list = []
    for stat in station_list:
        time_df, los_gps_data = load_gps_los_data(geo_path, station_name=stat, zero_start=False)
        df = time_df.to_frame()
        df['gps'] = los_gps_data
        df_list.append(df)
        # los_gps_list.append(los_gps_data)

        diff_df = pd.merge(ref_df, df, how='inner', on=['dt'])
        # diff_df['d_gps'] = diff_df['gps1'] - diff_df['gps2']
        gps_diff_ts = diff_df['gps'] - diff_df['ref_gps']
        gps_diff_ts = gps_diff_ts - np.mean(gps_diff_ts[:100])
        gps_dts = diff_df['dt']

    start_mean = np.mean(gps_diff_ts[:100])
    # start_mean = np.mean(los_gps_data2[:100])
    logger.info("Resetting GPS data start to 0 of station 2, subtract %f" % start_mean)
    gps_diff_ts = gps_diff_ts - start_mean

    plt.figure(fignum)
    plt.plot(gps_dts, gps_diff_ts, 'b.', label='gps data: %s' % stat1)

    days_smooth = 60
    los_gps_data_smooth = moving_average(gps_diff_ts, days_smooth)
    plt.plot(
        gps_dts,
        los_gps_data_smooth,
        'b',
        linewidth='4',
        # label='%d day smoothed gps data: %s' % (days_smooth, station_name))
        label='%d day smoothed gps data: %s-%s' % (days_smooth, stat1, stat2))

    insar_diff = insar_ts1 - insar_ts2

    plt.plot(geolist, insar_diff, 'r', label='insar data difference', ms=5)

    days_smooth = 5
    insar_diff_smooth = moving_average(insar_diff, days_smooth)
    insar_diff_smooth -= insar_diff_smooth[0]
    plt.plot(geolist,
             insar_diff_smooth,
             'r',
             label='linear %s day smoothed insar' % days_smooth,
             linewidth=3)

    # plt.ylim([-2, 2])
    plt.ylabel('cm of cumulative LOS displacement')
    plt.legend()

    return los_gps_data1, los_gps_data2, gps_diff_ts, insar_ts1, insar_ts2, insar_diff
