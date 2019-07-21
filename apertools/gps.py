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
    axes[0].plot(dts, _moving_average(east_cm, days_smooth), 'r-')
    axes[0].grid(True)
    remove_xticks(axes[0])

    axes[1].plot(dts, north_cm, 'b.')
    axes[1].set_ylabel('north (cm)')
    axes[1].plot(dts, _moving_average(north_cm, days_smooth), 'r-')
    axes[1].grid(True)
    remove_xticks(axes[1])

    axes[2].plot(dts, up_cm, 'b.')
    axes[2].set_ylabel('up (cm)')
    axes[2].plot(dts, _moving_average(up_cm, days_smooth), 'r-')
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


def difference_gps_stations(station1, station2):
    df1 = load_station_data(station1)
    df2 = load_station_data(station2)
    merged = pd.merge(df1, df2, how='inner', on=['dt'], suffixes=('_1', '_2'))
    merged['d_east'] = merged['east_1'] - merged['east_2']
    merged['d_north'] = merged['north_1'] - merged['north_2']
    merged['d_up'] = merged['up_1'] - merged['up_2']
    return merged
    # return merged[['dt', 'd_east', 'd_north', 'd_up']]


def _moving_average(arr, window_size=7):
    """Takes a 1D array and returns the running average of same size"""
    return uniform_filter1d(arr, size=window_size, mode='nearest')


def find_insar_ts(defo_filename='deformation.h5', station_list=[], window_size=1):
    """Get the insar timeseries closest to a list of GPS stations

    Returns the timeseries, and the datetimes of points for plotting

    Args:
        defo_filename
        station_list
        window_size (int): number of pixels in sqaure to average for insar timeseries
    """
    # geolist, deformation_stack = apertools.sario.load_deformation(full_path=defo_filename)
    # defo_img = apertools.latlon.load_deformation_img(full_path=defo_filename)

    insar_ts_list = []
    for station_name in station_list:
        lon, lat = station_lonlat(station_name)
        # row, col = defo_img.nearest_pixel(lat=lat, lon=lon)
        dem_rsc = apertools.sario.load_dem_from_h5(defo_filename)
        row, col = apertools.latlon.nearest_pixel(dem_rsc, lon=lon, lat=lat)
        insar_ts_list.append(get_stack_timeseries(defo_filename, row, col, window_size=window_size))

    geolist = apertools.sario.load_geolist_from_h5(defo_filename)
    return geolist, insar_ts_list


def get_stack_timeseries(filename,
                         row,
                         col,
                         stack_dset_name=apertools.sario.STACK_DSET,
                         station=None,
                         window_size=1):
    with h5py.File(filename) as f:
        try:
            return f[station][:]
        except KeyError:
            pass

    with h5py.File(filename, "a") as f:
        dset = f[stack_dset_name]
        ts = apertools.utils.window_stack(dset, row, col, window_size)
        if station is not None:
            f[station] = ts
        return ts


def _load_station_list(igrams_dir, defo_filename, station_name_list):
    defo_img = apertools.latlon.load_deformation_img(igrams_dir, filename=defo_filename)
    station_list = stations_within_image(defo_img)
    if station_name_list is not None:
        station_list = [s for s in station_list if s[0] in station_name_list]
    return station_list


def plot_gps_vs_insar(geo_path, defo_filename="deformation.npy", station_name_list=None):
    """Make a single plot of GPS vs InSAR
    Note that without the `align` option, the two might differ by an angle
    due to differences in reference points for the InSAR
    """
    igrams_dir = os.path.join(geo_path, 'igrams')
    station_list = _load_station_list(igrams_dir, defo_filename, station_name_list)

    for station_name, lon, lat in station_list:
        plt.figure()

        gps_dts, los_gps_data = load_gps_los_data(geo_path, station_name)

        plt.plot(gps_dts, los_gps_data, 'b.', label='gps data: %s' % station_name)

        days_smooth = 60
        los_gps_data_smooth = _moving_average(los_gps_data, days_smooth)
        plt.plot(gps_dts,
                 los_gps_data_smooth,
                 'b',
                 linewidth='4',
                 label='%d day smoothed gps data: %s' % (days_smooth, station_name))

        geolist, insar_ts = find_insar_ts(defo_filename=defo_filename, station_list=[station_name])

        plt.plot(geolist, insar_ts, 'rx', label='insar data', ms=5)

        days_smooth = 5
        insar_ts_smooth = _moving_average(insar_ts, days_smooth)
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
    # station_list = list(reversed(_load_station_list(igrams_dir, defo_filename, station_name_list)))
    station_list = _load_station_list(igrams_dir, defo_filename, station_name_list)

    stat1, lon1, lat1 = station_list[0]
    stat2, lon2, lat2 = station_list[1]
    logger.info("Using stations: %s" % station_list[:2])

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
    los_gps_data_smooth = _moving_average(gps_diff_ts, days_smooth)
    plt.plot(
        gps_dts,
        los_gps_data_smooth,
        'b',
        linewidth='4',
        # label='%d day smoothed gps data: %s' % (days_smooth, station_name))
        label='%d day smoothed gps data: %s-%s' % (days_smooth, stat1, stat2))

    geolist, (insar_ts1, insar_ts2) = find_insar_ts(defo_filename=defo_filename,
                                                    station_list=[stat1, stat1])
    insar_diff = insar_ts1 - insar_ts2

    plt.plot(geolist, insar_diff, 'r', label='insar data difference', ms=5)

    days_smooth = 5
    insar_diff_smooth = _moving_average(insar_diff, days_smooth)
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


def plot_all_gps_insar(geo_path,
                       fignum=None,
                       defo_filename='deformation.h5',
                       ref_station=None,
                       station_name_list=None):
    """
    For a subset, pass in the station_name_list
    """
    igrams_dir = os.path.join(geo_path, 'igrams')
    station_list = _load_station_list(igrams_dir, defo_filename, station_name_list)

    ref_dts, ref_los_gps_data = load_gps_los_data(geo_path, ref_station, zero_start=False)
    ref_df = ref_dts.to_frame()
    ref_df['ref_gps'] = ref_los_gps_data

    # First way: project each ENU to LOS, then subtract
    # gps_dt_list, los_gps_list = [], []
    df_list = []
    for stat in station_list:
        time_df, los_gps_data = load_gps_los_data(geo_path, stat, zero_start=False)
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
    los_gps_data_smooth = _moving_average(gps_diff_ts, days_smooth)
    plt.plot(
        gps_dts,
        los_gps_data_smooth,
        'b',
        linewidth='4',
        # label='%d day smoothed gps data: %s' % (days_smooth, station_name))
        label='%d day smoothed gps data: %s-%s' % (days_smooth, stat1, stat2))

    geolist, (insar_ts1, insar_ts2) = find_insar_ts(defo_filename=defo_filename,
                                                    station_list=[stat1, stat1])
    insar_diff = insar_ts1 - insar_ts2

    plt.plot(geolist, insar_diff, 'r', label='insar data difference', ms=5)

    days_smooth = 5
    insar_diff_smooth = _moving_average(insar_diff, days_smooth)
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
