"""gps.py
Utilities for integrating GPS with InSAR maps

Links:
list of LLH for all gps stations: ftp://gneiss.nbmg.unr.edu/rapids/llh
Clickable names to explore: http://geodesy.unr.edu/NGLStationPages/GlobalStationList
Map of stations: http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html
"""
from __future__ import division, print_function
import os
import glob
import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

import apertools
from apertools.log import get_log

logger = get_log()

# URL for ascii file of 24-hour final GPS solutions in east-north-vertical (NA12)
GPS_BASE_URL = "http://geodesy.unr.edu/gps_timeseries/tenv3/NA12/{station}.NA12.tenv3"

GPS_DIR = os.environ.get('GPS_DIR', '/data1/scott/pecos/gps_station_data')

# STATION_LLH_FILE = "texas_stations.csv"
STATION_LLH_FILE = "station_llh_all.csv"


def load_station_data(station_name,
                      gps_dir=GPS_DIR,
                      start_year=2015,
                      end_year=2018,
                      download_if_missing=True):
    """Loads one gps station's ENU data since start_year until end_year

    Args:
        station_name (str): 4 Letter name of GPS station
            See http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html for map
        gps_dir (str): directory containing gps station lla csv for read_station_llas
        start_year (int), default 2015, cutoff for beginning of GPS data
        end_year (int): default 2018, cut off for end of GPS data
        download_if_missing (bool): default True
    """
    gps_data_file = os.path.join(gps_dir, '%s.NA12.tenv3' % station_name)
    if not os.path.exists(gps_data_file):
        logger.warning("%s does not exist.", gps_data_file)
        if download_if_missing:
            logger.info("Downloading %s", station_name)
            download_station_data(station_name, gps_dir=gps_dir)

    df = pd.read_csv(gps_data_file, header=0, sep=r"\s+")
    return _clean_gps_df(df, start_year, end_year)


def stations_within_image(image_ll=None,
                          image_filename=None,
                          mask_invalid=True,
                          gps_dir=None,
                          gps_filename=None):
    """Given an image, find gps stations contained in area

    Must have an associated .rsc file, either from being a LatlonImage (passed
    by image_ll), or have the file in same directory as `image_filename`

    Args:
        image_ll (LatlonImage): LatlonImage of area with data
        image_filename (str): filename to load into a LatlonImage
        mask_invalid (bool): Default true. if true, don't return stations
            where the image value is NaN or exactly 0
        gps_dir (str): directory containing gps station lla csv for read_station_llas

    Returns:
        ndarray: Nx3, with columns ['name', 'lon', 'lat']
    """
    if image_ll is None:
        image_ll = apertools.latlon.LatlonImage(filename=image_filename)

    df = read_station_llas(gps_dir=gps_dir, filename=gps_filename)
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


def plot_stations(image_ll=None, image_filename=None, mask_invalid=True):
    """Plot an GPS station points contained within an image"""
    if image_ll is None:
        try:
            # First check if we want to load the 3D stack "deformation.npy"
            image_ll = apertools.latlon.load_deformation_img(filename=image_filename)
        except ValueError:
            image_ll = apertools.latlon.LatlonImage(filename=image_filename)

    stations = stations_within_image(image_ll, mask_invalid=mask_invalid)
    # TODO: maybe just plot_image_shifted
    fig, ax = plt.subplots()
    axim = ax.imshow(image_ll, extent=image_ll.extent)
    fig.colorbar(axim)

    for name, lon, lat in stations:
        ax.plot(lon, lat, 'X', markersize=15, label=name)
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
def read_station_llas(gps_dir=None, header=None, filename=None):
    """Read in the name, lat, lon, alt list of gps stations

    Assumes file is a csv with "name,lat,lon,alt" as columns
    Must give "header" argument if there is a header
    """
    gps_dir = gps_dir or GPS_DIR
    filename = filename or STATION_LLH_FILE
    full_filename = os.path.join(GPS_DIR, filename)
    logger.info("Searching %s for gps data" % full_filename)
    df = pd.read_csv(full_filename, header=header)
    df.columns = ['name', 'lat', 'lon', 'alt']
    return df


def station_lonlat(station_name, gps_dir=None):
    """Return the (lon, lat) of a `station_name`"""
    df = read_station_llas(gps_dir=gps_dir)
    name, lat, lon, alt = df[df['name'] == station_name].iloc[0]
    return lon, lat


def stations_within_rsc(rsc_filename=None, rsc_data=None, gps_dir=None, gps_filename=None):
    if rsc_data is None:
        if rsc_filename is None:
            raise ValueError("Need rsc_data or rsc_filename")
        rsc_data = apertools.sario.load(rsc_filename)

    df = read_station_llas(gps_dir=gps_dir, filename=gps_filename)
    station_lon_lat_arr = df[['lon', 'lat']].values
    contains_bools = [apertools.latlon.grid_contains(s, **rsc_data) for s in station_lon_lat_arr]
    return df[contains_bools][['name', 'lon', 'lat']].values


def _clean_gps_df(df, start_year, end_year=None):
    df['dt'] = pd.to_datetime(df['YYMMMDD'], format='%y%b%d')

    df_ranged = df[df['dt'] >= datetime.datetime(start_year, 1, 1)]
    if end_year:
        df_ranged = df_ranged[df_ranged['dt'] <= datetime.datetime(end_year, 1, 1)]
    df_enu = df_ranged[['dt', '__east(m)', '_north(m)', '____up(m)']]
    df_enu = df_enu.rename(mapper=lambda s: s.replace('_', '').replace('(m)', ''), axis='columns')
    return df_enu


def download_station_data(station_name, gps_dir=None):
    url = GPS_BASE_URL.format(station=station_name)
    response = requests.get(url)
    if gps_dir is None:
        gps_dir = GPS_DIR

    stationfile = url.split("/")[-1]  # Just blah.NA12.tenv3
    filename = "{}/{}".format(gps_dir, stationfile)
    logger.info("Saving to {}".format(filename))

    with open(filename, "w") as f:
        f.write(response.text)


def plot_gps_enu(station=None, days_smooth=12):
    """Plot the east,north,up components of `station`"""

    def remove_xticks(ax):
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

    station_df = load_station_data(station)

    fig, axes = plt.subplots(3, 1)
    east_mm = 100 * (station_df['east'] - station_df['east'].iloc[0])
    north_mm = 100 * (station_df['north'] - station_df['north'].iloc[0])
    up_mm = 100 * (station_df['up'] - station_df['up'].iloc[0])
    dts = station_df['dt']
    axes[0].plot(dts, east_mm, 'b.')
    axes[0].set_ylabel('east (cm)')
    axes[0].plot(dts, _moving_average(east_mm, days_smooth), 'r-')
    axes[0].grid(True)
    remove_xticks(axes[0])

    axes[1].plot(dts, north_mm, 'b.')
    axes[1].set_ylabel('north (cm)')
    axes[1].plot(dts, _moving_average(north_mm, days_smooth), 'r-')
    axes[1].grid(True)
    remove_xticks(axes[1])

    axes[2].plot(dts, up_mm, 'b.')
    axes[2].set_ylabel('up (cm)')
    axes[2].plot(dts, _moving_average(up_mm, days_smooth), 'r-')
    axes[2].grid(True)
    remove_xticks(axes[2])

    fig.suptitle(station)

    return fig, axes


def load_gps_los_data(insar_dir, station_name=None, to_cm=True, zero_start=True):
    """Load the GPS timeseries of a station name

    Returns the timeseries, and the datetimes of the points
    """
    lon, lat = station_lonlat(station_name)
    enu_coeffs = apertools.los.find_enu_coeffs(lon, lat, insar_dir)

    df = load_station_data(station_name)
    enu_data = df[['east', 'north', 'up']].T
    los_gps_data = apertools.los.project_enu_to_los(enu_data, enu_coeffs=enu_coeffs)

    if to_cm:
        logger.info("Converting GPS to cm:")
        los_gps_data = 100 * los_gps_data

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


def find_insar_ts(insar_dir, station_name, defo_name='deformation.npy'):
    """Get the insar timeseries closest to a GPS station

    Returns the timeseries, and the datetimes of points for plotting
    """
    igrams_dir = os.path.join(insar_dir, 'igrams')
    geolist, deformation_stack = apertools.sario.load_deformation(igrams_dir, filename=defo_name)
    defo_img = apertools.latlon.load_deformation_img(igrams_dir, filename=defo_name)

    lon, lat = station_lonlat(station_name)
    insar_row, insar_col = defo_img.nearest_pixel(lat=lat, lon=lon)
    # import pdb
    # pdb.set_trace()
    insar_ts = apertools.utils.window_stack(deformation_stack,
                                            insar_row,
                                            insar_col,
                                            window_size=5,
                                            func=np.mean)
    return geolist, insar_ts


def plot_gps_vs_insar():
    insar_dir = '/data1/scott/pecos/path85/N31.4W103.7'

    # station_name = 'TXKM'
    # los_dir = '/data4/scott/delaware-basin/test2/N31.4W103.7/'
    # lla, xyz = los.read_los_output(os.path.join(los_dir, 'extra_files/los_vectors.txt'))
    # enu_coeffs = los.find_enu_coeffs(-102.894010019, 31.557733084, insar_dir)

    # lon, lat = station_lonlat(station_name)
    # df = load_station_data(station_name)
    # enu_data = df[['east', 'north', 'up']].T
    # los_gps_data = project_enu_to_los(enu_data, los_vec, lat, lon)
    # los_gps_data = los.project_enu_to_los(enu_data, enu_coeffs=enu_coeffs)

    defo_name = 'deformation.npy'
    igrams_dir = os.path.join(insar_dir, 'igrams')
    defo_img = apertools.latlon.load_deformation_img(igrams_dir, filename=defo_name)
    station_list = stations_within_image(defo_img)
    for station_name, lon, lat in station_list:
        plt.figure()

        gps_dts, los_gps_data = load_gps_los_data(insar_dir, station_name)

        plt.plot(gps_dts, los_gps_data, 'b.', label='gps data: %s' % station_name)

        days_smooth = 60
        los_gps_data_smooth = _moving_average(los_gps_data, days_smooth)
        plt.plot(gps_dts,
                 los_gps_data_smooth,
                 'b',
                 linewidth='4',
                 label='%d day smoothed gps data: %s' % (days_smooth, station_name))

        geolist, insar_ts = find_insar_ts(insar_dir, station_name, defo_name=defo_name)

        plt.plot(geolist, insar_ts, 'rx', label='insar data', ms=5)

        days_smooth = 5
        insar_ts_smooth = _moving_average(insar_ts, days_smooth)
        plt.plot(geolist,
                 insar_ts_smooth,
                 'r',
                 label='%s day smoothed insar' % days_smooth,
                 linewidth=3)

        plt.legend()
    # return geolist, insar_ts, gps_dts, los_gps_data, defo_img


def plot_gps_vs_insar_diff(fignum=None, defo_name='deformation.npy'):
    insar_dir = '/data1/scott/pecos/path85/N31.4W103.7'

    igrams_dir = os.path.join(insar_dir, 'igrams')
    defo_img = apertools.latlon.load_deformation_img(igrams_dir, filename=defo_name)
    station_list = stations_within_image(defo_img)

    stat1, lon1, lat1 = station_list[0]
    stat2, lon2, lat2 = station_list[1]

    # # First way: project each ENU to LOS, then subtract
    gps_dts1, los_gps_data1 = load_gps_los_data(insar_dir, stat1, zero_start=False)
    gps_dts2, los_gps_data2 = load_gps_los_data(insar_dir, stat2, zero_start=False)
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

    # # Other way: subtracting ENU components, then project diffs to LOS
    # enu_coeffs = los.find_enu_coeffs(lon1, lat1, insar_dir)
    # # Check how different they are
    # enu_coeffs2 = los.find_enu_coeffs(lon2, lat2, insar_dir)
    # import pdb
    # pdb.set_trace()
    # diff_df = difference_gps_stations(stat1, stat2)
    # enu_data = diff_df[['d_east', 'd_north', 'd_up']].T
    # gps_diff_ts = los.project_enu_to_los(enu_data, enu_coeffs=enu_coeffs)
    # gps_dts = diff_df['dt']

    # logger.info("Converting GPS to cm:")
    # gps_diff_ts = 100 * gps_diff_ts

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

    geolist, insar_ts1 = find_insar_ts(insar_dir, stat1, defo_name=defo_name)
    _, insar_ts2 = find_insar_ts(insar_dir, stat2, defo_name=defo_name)
    insar_diff = insar_ts1 - insar_ts2

    plt.plot(geolist, insar_diff, 'r', label='insar data difference', ms=5)

    days_smooth = 5
    insar_diff_smooth = _moving_average(insar_diff, days_smooth)
    plt.plot(geolist,
             insar_diff_smooth,
             'r',
             label='linear %s day smoothed insar' % days_smooth,
             linewidth=3)

    plt.ylim([-2, 2])
    plt.ylabel('cm of cumulative LOS displacement')
    plt.legend()
    return los_gps_data1, los_gps_data2, gps_diff_ts, insar_ts1, insar_ts2, insar_diff


def find_stations_with_data(gps_dir=None):
    """
        gps_dir (str): directory containing gps station lla csv for read_station_llas
    """
    # Now also get gps station list
    if not gps_dir:
        gps_dir = GPS_DIR

    all_station_data = read_station_llas(gps_dir=gps_dir)
    station_data_list = find_station_data_files(gps_dir)
    stations_with_data = [
        tup for tup in all_station_data.to_records(index=False) if tup[0] in station_data_list
    ]
    return stations_with_data


def find_station_data_files(gps_dir):
    station_files = glob.glob(os.path.join(gps_dir, '*.tenv3'))
    station_list = []
    for filename in station_files:
        _, name = os.path.split(filename)
        station_list.append(name.split('.')[0])
    return station_list


# def read_station_dict(filename):
#     """Reads in GPS station data"""
#     with open(filename) as f:
#         station_strings = [row for row in f.read().splitlines()]
#
#     all_station_data = []
#     for row in station_strings:
#         name, lat, lon, _ = row.split(',')  # Ignore altitude
#         all_station_data.append((name, float(lon), float(lat)))
#     return all_station_data

# def gps_to_los():
#     insar_dir = '/data4/scott/delaware-basin/test2/N31.4W103.7'
#     lla, xyz = los.read_los_output(os.path.join(insar_dir, 'extra_files/los_vectors.txt'))
#
#     los_vec = np.array(xyz)[0]
#
#     station_name = 'TXKM'
#     df = load_station_data(station_name)
#     lon, lat = station_lonlat(station_name)
#     enu_data = df[['east', 'north', 'up']].T
#     los_gps_data = los.project_enu_to_los(enu_data, los_vec, lat, lon)
#     return los_gps_data, df['dt']
