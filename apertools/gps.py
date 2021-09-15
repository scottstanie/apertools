"""gps.py
Utilities for integrating GPS with InSAR maps

Links:

1. list of LLH for all gps stations: ftp://gneiss.nbmg.unr.edu/rapids/llh
Note: ^^ This file is stored in the `STATION_LLH_FILE`

2. Clickable names to explore: http://geodesy.unr.edu/NGLStationPages/GlobalStationList
3. Map of stations: http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html

"""
from __future__ import division, print_function
from glob import glob
import re
import os
import difflib  # For station name misspelling checks
import datetime
from dataclasses import dataclass
from functools import lru_cache

import requests
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.dates as mdates

import apertools.latlon
import apertools.los
import apertools.utils
import apertools.sario
from apertools.sario import LOS_FILENAME
import apertools.plotting
from apertools.log import get_log

logger = get_log()

# URL for ascii file of 24-hour final GPS solutions in east-north-vertical (NA12)
# GPS_BASE_URL = "http://geodesy.unr.edu/gps_timeseries/tenv3/NA12/{station}.NA12.tenv3"
# UPDATE 4/20/20: noticed they changed it to
GPS_BASE_URL = (
    "http://geodesy.unr.edu/gps_timeseries/tenv3/plates/{plate}/{station}.{plate}.tenv3"
)
# NOTE: for now i'm assuming all plate-fixed data is the only one i care about...
# if i also want IGS14, i'll need to divide directories and do more
GPS_FILE = GPS_BASE_URL.split("/")[-1].replace(".{plate}", "")
# The main web page per station
# We'll use this for now to scrape the plate information with a regex :(
GPS_STATION_URL = "http://geodesy.unr.edu/NGLStationPages/stations/{station}.sta"

DIRNAME = os.path.dirname(os.path.abspath(__file__))


def _get_gps_dir():
    path = apertools.utils.get_cache_dir(force_posix=True)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


GPS_DIR = _get_gps_dir()

# These lists get update occasionally... to keep fresh, download one for current day
# old ones will be removed upon new download
STATION_LLH_URL = "http://geodesy.unr.edu/NGLStationPages/llh.out"
STATION_LLH_FILE = os.path.join(GPS_DIR, "station_llh_all_{today}.csv")

STATION_XYZ_URL = "http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
STATION_XYZ_FILE = os.path.join(GPS_DIR, "station_xyz_all_{today}.csv")


@dataclass
class InsarGPSCompare:
    insar_filename: str = "deformation.h5"
    dset: str = "linear_velocity"
    los_map_file: str = LOS_FILENAME
    # to measure GPS relative to some other station, set the reference station
    reference_station: str = None
    # Used to average InSAR values in a box around the stations
    window_size: int = 3
    # These will get used by in the GPS df creation; they're set using the InSAR data
    start_date: datetime.date = None
    end_date: datetime.date = None
    # To limit stations that have at least 60% coverage over the
    # time period we care about, set a nan threshold of 0.4
    max_nan_pct: float = 0.4
    # To smooth the GPS or insar timeseries, set the number of smoothing days
    days_smooth_insar: int = None
    days_smooth_gps: int = None
    # Create an extra column the the output with the difference
    # of (GPS - InSAR) for each station
    create_diffs: bool = True
    # Use the median trend to compare differences
    median: bool = True
    # convert velocity comparisons to millimeter/year
    to_mm_year: bool = True
    print_summary: bool = True

    def run(self):
        """Create the GPS/InSAR DataFrame and compare average velocities

        Returns:
            df_full (DataFrame) with 1 GPS, 1 InSAR, and 1 _diff column per station
            df_velo_diffs (DataFrame): 3-column dataframe comparing average velocities.
                Columns: velo_diff     v_gps   v_insar
                Each row is one station
        """
        df_full = self.create_df()
        df_velo_diffs = self.compare_velocities(
            to_mm_year=self.to_mm_year, print_summary=self.print_summary
        )
        return df_full, df_velo_diffs

    def create_df(self):
        """Set days_smooth to None or 0 to avoid any data smoothing

        If refernce station is specified, all timeseries will subtract
        that station's data
        """
        df_gps_locations = get_stations_within_image(
            filename=self.insar_filename, dset=self.dset
        )

        df_insar = self.create_insar_df(df_gps_locations)
        # Cap the GPS we use by the InSAR start/end dates
        self._set_start_end_date(df_insar)
        df_gps = self.create_gps_los_df(df_gps_locations)
        df = self.combine_insar_gps_dfs(df_insar, df_gps)
        if self.reference_station is not None:
            df = self._subtract_reference(df)
        if self.create_diffs:
            for stat in df_gps_locations["name"]:
                df[f"{stat}_diff"] = df[f"{stat}_gps"] - df[f"{stat}_insar"]

        self.df = self._remove_bad_cols(df)
        return self.df

    def compare_velocities(self, median=None, to_mm_year=True, print_summary=True):
        """Given the combine insar/gps dataframe, fit and compare slopes

        Args:
            median (bool): optional. If True, uses TSIA median estimator
            to_mm_year (bool): convert the velocities to mm/year. Otherwise, cm/day
        """
        df = getattr(self, "df", None)
        if df is None:
            raise ValueError("Must run create_df before compare_velocities")
        if median is None:
            median = self.median

        station_names = self.get_stations()
        df_velo_diffs = pd.DataFrame({"name": station_names})
        diffs = []
        v_gps_list = []
        v_insar_list = []
        for station in station_names:
            ts_gps = df[station + "_gps"]
            ts_insar = df[station + "_insar"]
            v_gps = fit_line(ts_gps, median=self.median)[0]
            v_insar = fit_line(ts_insar, median=self.median)[0]
            v_gps_list.append(v_gps)
            v_insar_list.append(v_insar)
            diffs.append(v_gps - v_insar)
        df_velo_diffs["velo_diff"] = diffs
        df_velo_diffs["v_gps"] = v_gps_list
        df_velo_diffs["v_insar"] = v_insar_list
        if to_mm_year:  # as opposed to cm/day
            df_velo_diffs[["velo_diff", "v_gps", "v_insar"]] *= 365.25 * 10
        df_velo_diffs.set_index("name", inplace=True)
        if print_summary:
            units = "mm/year" if to_mm_year else "cm/day"
            print("RMS Difference:")
            print(f"{self.rms(df_velo_diffs['velo_diff']):.3f} {units}")
        self.df_velo_diffs = df_velo_diffs
        return df_velo_diffs

    def get_stations(self):
        """Takes df with columns ['NMHB_insar', 'TXAD_insar',...],
        returns list of unique station names"""
        # Check that we have run `create_df`
        df = getattr(self, "df", None)
        if df is None:
            return []
        return list(sorted(set(map(lambda s: s.split("_")[0], df.columns))))

    def create_insar_df(self, df_gps_locations):
        """Set days_smooth to None or 0 to avoid any data smoothing"""
        with xr.open_dataset(self.insar_filename) as ds:
            da = ds[self.dset]
            if "date" in da.coords:
                # 3D option
                is_2d = False
                dates = da.indexes["date"]
            else:
                # 2D: just the average ground velocity
                is_2d = True
                dates = ds.indexes["date"]
        df_insar = pd.DataFrame({"date": dates})
        for row in df_gps_locations.itertuples():
            ts = apertools.utils.window_stack_xr(
                da, lon=row.lon, lat=row.lat, window_size=self.window_size
            )
            if is_2d:
                # Make a cum_defo from the linear trend
                x = (dates - dates[0]).days
                v_cm_yr = ts.item()
                coeffs = [v_cm_yr / 365.25, 0]
                df_insar[row.name] = linear_trend(coeffs=coeffs, x=x)
                # NOTE: To recover the linear velocity used here:
                # gps.fit_line(df_insar[station_name])[0] * 365.25
            else:
                df_insar[row.name] = ts

        df_insar.set_index("date", inplace=True)
        self.df_insar = df_insar
        return df_insar

    def create_gps_los_df(self, df_gps_locations, start_date=None, end_date=None):
        return self._create_gps_df(
            df_gps_locations, kind="los", start_date=start_date, end_date=end_date
        )

    def create_gps_enu_df(self, df_gps_locations, start_date=None, end_date=None):
        return self._create_gps_df(
            df_gps_locations, kind="enu", start_date=start_date, end_date=end_date
        )

    def _create_gps_df(
        self, df_gps_locations, start_date=None, end_date=None, kind="los"
    ):
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        df_list = []
        # df_locations = get_stations_within_image(filename=los_map_file)
        for row in df_gps_locations.itertuples():
            if kind.lower() == "los":
                df_los = load_gps_los(
                    los_map_file=self.los_map_file,
                    station_name=row.name,
                    start_date=start_date,
                    end_date=end_date,
                    zero_start=True,
                    # coordinates=self.coordinates,
                )
            elif kind.lower() == "enu":
                df_los = load_station_enu(
                    station_name=row.name,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )

            # np.array(pd.Series(arr).rolling(window_size).mean())
            # df = pd.DataFrame({"date": _series_to_date(gps_dts)})
            # df[stat + "_gps"] = moving_average(los_gps_data, days_smooth)
            # df[stat + "_smooth_gps"] = moving_average(los_gps_data, days_smooth)
            df_list.append(df_los)
        # Now merge each one in turn, keeping all dates
        df_gps = pd.concat(df_list, join="outer", axis="columns")
        # These will all have the same name, so set equal to the station
        df_gps.columns = df_gps_locations.name.values
        self.df_gps = df_gps
        return df_gps

    def _set_start_end_date(self, df_insar):
        # constrain the date range to just the InSAR min/max dates
        self.start_date = df_insar.index.min()
        self.end_date = df_insar.index.max()

    def combine_insar_gps_dfs(self, df_insar, df_gps):
        df = pd.merge(
            df_gps,
            df_insar,
            how="left",
            left_on="date",
            right_on="date",
            suffixes=("_gps", "_insar"),
        )
        if self.start_date:
            df = df.loc[(df.index >= self.start_date)]
        if self.end_date:
            df = df.loc[(df.index <= self.end_date)]
        return df

    def rms(self, errors=None):
        if errors is None:
            errors = self.df_velo_diffs["velo_diffs"]
        return np.sqrt(np.mean(np.square(errors)))

    def total_abs_error(self, errors):
        if errors is None:
            errors = self.df_velo_diffs["velo_diffs"]
        return np.sum(np.abs(errors))

    def _find_bad_cols(
        self, df, max_nan_pct=0.4, empty_start_len=None, empty_end_len=None
    ):
        # If we care about and empty `empty_start_len` entries at the beginnning, make into int
        empty_starts = df.columns[df.head(empty_start_len).isna().all()]
        if empty_end_len:
            empty_ends = df.columns[df.tail(empty_end_len).isna().all()]
        else:
            empty_ends = []
        nan_pcts = df.isna().sum() / len(df)
        # print("nan pcts:\n", nan_pcts)
        high_pct_nan = df.columns[nan_pcts > max_nan_pct]
        # Ignore all the insar nans
        high_pct_nan = [c for c in high_pct_nan if "gps" in c]
        all_cols = np.concatenate(
            (
                np.array(empty_starts),
                np.array(empty_ends),
                np.array(high_pct_nan),
            )
        )
        return list(set(all_cols))

    def _remove_bad_cols(self, df):
        """Drops columns that are all NaNs, or where GPS doesn't cover whole period"""
        bad_cols = self._find_bad_cols(df, max_nan_pct=self.max_nan_pct)
        logger.info("Removing the following bad columns:")
        logger.info(bad_cols)

        df_out = df.copy()
        for col in bad_cols:
            if col not in df_out.columns:
                continue
            df_out.drop(col, axis=1, inplace=True)
            station = col.replace("_gps", "").replace("_insar", "")
            c = "%s_gps" % station
            if c in df_out.columns:
                df_out.drop(c, axis=1, inplace=True)
            c = "%s_insar" % station
            if c in df_out.columns:
                df_out.drop(c, axis=1, inplace=True)
            c = "%s_diff" % station
            if c in df_out.columns:
                df_out.drop(c, axis=1, inplace=True)
        return df_out

    def _subtract_reference(self, df):
        """Center all columns of `df` based on the `reference_station` columns"""
        gps_ref_col = "%s_%s" % (self.reference_station, "gps")
        insar_ref_col = "%s_%s" % (self.reference_station, "insar")
        df_out = df.copy()
        for col in df.columns:
            if "gps" in col:
                df_out[col] = df[col] - df[gps_ref_col]
            elif "insar" in col:
                df_out[col] = df[col] - df[insar_ref_col]
        return df_out


@dataclass
class TrendEstimator:
    series: pd.Series
    tol_days: int = 30

    def tsia(self):
        """Calculate the Thiel-Sen Inter-annual slope of a data Series

        https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator

        Assumes the `series` has a DatetimeIndex.
        Forms all possible difference of data which span 1 year +/- `tol_days`,
        then takes the median slope of these differences
        """
        # Get the non-nan values of the series
        data = self.series.dropna().values
        times = self.series.dropna().index
        # Convert to numerical values for fitting:
        # t = (times - times[0]).days
        t = mdates.date2num(times)
        time_diffs = self._get_all_differences(t)
        slopes = self._get_all_differences(data) / time_diffs

        # Now pick slopes within `tol_days` of annual
        # > 180 to make sure we dont' use super short periods
        accept_idxs = np.logical_and(
            time_diffs > 180, (self._dist_from_year(time_diffs) < self.tol_days)
        )
        slopes_annual = slopes[accept_idxs]
        slope = np.median(slopes_annual)

        # Add Normal dist. factor to MAD
        sig = 1.4826 * self.mad(slopes_annual)
        # TODO: track down Ben for origina of this formula... prob on Wiki for TSIA
        uncertainty = 3 * np.sqrt(np.pi / 2) * sig / np.sqrt(len(slopes_annual) / 4)
        b = np.median(data - slope * t)
        return slope, b, uncertainty

    @staticmethod
    def mad(x):
        """Median absolut deviation"""
        return np.median(np.abs(x - np.median(x)))

    @staticmethod
    def _dist_from_year(v):
        """Get the number of days away from 365, mod 1 year"""
        return np.abs((v + 180) % 365 - 180)

    @staticmethod
    def _get_all_differences(a):
        """Calculate all possible differences between elements of `a`"""
        n = len(a)
        x = np.reshape(a, (1, n))
        difference_matrix = x - x.transpose()
        # Now get the upper half (bottom is redundant)
        return difference_matrix[np.triu_indices(n)].ravel()


def get_final_east_values(east_df):
    stations, vals = [], []

    direc = None
    for column, val in east_df.tail(14).mean().items():
        station, d = column.split("_")
        direc = d
        stations.append(station)
        vals.append(val)
    return pd.DataFrame(index=stations, data={direc: vals})


def fit_line(series, median=False):
    """Fit a line to `series` with (possibly) uneven dates as index.

    Can be used to detrend, or predict final value

    Args:
        series (pd.Series): data to fit, with a DatetimeIndex
        median (bool): if true, use the TSIA median estimator to fit

    Returns: [slope, intercept]
    """
    # TODO: check that subtracting first item doesn't change it

    series_clean = series.dropna()
    idxs = mdates.date2num(series_clean.index)

    coeffs = np.polyfit(idxs, series_clean, 1)
    if median:
        # Replace the Least squares fit with the median inter-annual slope
        est = TrendEstimator(series)
        # med_slope, intercept, uncertainty = est.tsia()
        coeffs = est.tsia()[:2]
    return coeffs


def linear_trend(series=None, coeffs=None, index=None, x=None, median=False):
    """Get a series of points representing a linear trend through `series`

    First computes the lienar regression, the evaluates at each
    dates of `series.index`

    Args:
        series (pandas.Series): data with DatetimeIndex as the index.
        coeffs (array or List): [slope, intercept], result from np.polyfit
        index (DatetimeIndex, list[date]): Optional. If not passing series, can pass
            the DatetimeIndex or list of dates to evaluate coeffs at.
            Converts to numbers using `matplotlib.dates.date2num`
        x (ndarray-like): directly pass the points to evaluate the poly1d
    Returns:
        Series: a line, equal length to arr, with same index as `series`
    """
    if coeffs is None:
        coeffs = fit_line(series, median=median)

    if index is None and x is None:
        index = series.dropna().index
    if x is None:
        x = mdates.date2num(index)

    poly = np.poly1d(coeffs)
    linear_points = poly(x)
    return pd.Series(linear_points, index=index)


def _flat_std(series):
    """Find the std dev of an Series with a linear component removed"""
    return np.std(series - linear_trend(series))


def load_station_enu(
    station_name,
    start_date=None,
    end_date=None,
    download_if_missing=True,
    force_download=False,
    zero_by="mean",
    to_cm=True,
):
    """Loads one gps station's ENU data since start_date until end_date as a dataframe

    Args:
        station_name (str): 4 Letter name of GPS station
            See http://geodesy.unr.edu/NGLStationPages/gpsnetmap/GPSNetMap.html for map
        start_date (datetime or str): Optional. cutoff for beginning of GPS data
        end_date (datetime or str): Optional. cut off for end of GPS data
        download_if_missing (bool): default True
        force_download (bool): default False
    """
    # start_date, end_date = _parse_dates(start_date, end_date)
    if zero_by not in ("start", "mean"):
        raise ValueError("'zero_by' must be either 'start' or 'mean'")
    station_name = station_name.upper()
    gps_data_file = os.path.join(GPS_DIR, GPS_FILE.format(station=station_name))
    if force_download:
        try:
            os.remove(gps_data_file)
            logger.info(f"force removed {gps_data_file}")
        except FileNotFoundError:
            pass
    if not os.path.exists(gps_data_file):
        if download_if_missing:
            logger.info(f"Downloading {station_name} to {gps_data_file}")
            download_station_data(station_name)
        else:
            raise ValueError(
                "{gps_data_file} does not exist, download_if_missing = False"
            )

    df = pd.read_csv(gps_data_file, header=0, sep=r"\s+", engine="c")
    clean_df = _clean_gps_df(df, start_date, end_date)
    if to_cm:
        # logger.info("Converting %s GPS to cm" % station_name)
        clean_df[["east", "north", "up"]] = 100 * clean_df[["east", "north", "up"]]

    if zero_by.lower() == "mean":
        mean_val = clean_df[["east", "north", "up"]].mean()
        # enu_zeroed = clean_df[["east", "north", "up"]] - mean_val
        clean_df[["east", "north", "up"]] -= mean_val
    elif zero_by.lower() == "start":
        start_val = clean_df[["east", "north", "up"]].iloc[:10].mean()
        # enu_zeroed = clean_df[["east", "north", "up"]] - start_val
        clean_df[["east", "north", "up"]] -= start_val
    # Finally, make the 'date' column a DateIndex
    return clean_df.set_index("date")


def _clean_gps_df(df, start_date=None, end_date=None):
    """Reorganize the Nevada GPS data format"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["YYMMMDD"], format="%y%b%d")

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    df_enu = df[["date", "__east(m)", "_north(m)", "____up(m)"]]
    df_enu = df_enu.rename(
        mapper=lambda s: s.replace("_", "").replace("(m)", ""), axis="columns"
    )
    df_enu.reset_index(inplace=True, drop=True)
    return df_enu


def get_stations_within_image(
    filename=None,
    dset=None,
    da=None,
    bad_vals=[0],
    mask_invalid=True,
):
    """Given an image, find gps stations contained in area

    Should be GDAL- or xarray-readable with lat/lon coordinates

    Args:
        filename (str): filename to load
        mask_invalid (bool): Default true. if true, don't return stations
            where the image value is NaN or exactly 0
        bad_vals (list[float]): values (beside nan) indicating no data
            (default: [0])

    Returns:
        ndarray: Nx3, with columns ['name', 'lon', 'lat']
    """
    from shapely import geometry

    if da is None:
        try:
            da = xr.open_dataset(filename)[dset]
        except Exception:
            import rioxarray

            da = rioxarray.open_rasterio(filename).rename({"x": "lon", "y": "lat"})

    # Do i care to filter out those not really in the image for radar coords?
    is_2d_latlon = da.lat.ndim == 2
    gdf = read_station_llas(to_geodataframe=True)
    image_bbox = geometry.box(*apertools.latlon.bbox_xr(da))
    gdf_within = gdf[gdf.geometry.within(image_bbox)]
    # good_stations = []
    # Will need to select differently for radar coords
    if mask_invalid:
        good_idxs = []
        for row in gdf_within.itertuples():
            if is_2d_latlon:
                r, c = apertools.latlon.latlon_to_rowcol_rdr(
                    row.lat,
                    row.lon,
                    lat_arr=da.lat.data,
                    lon_arr=da.lon.data,
                    warn_oob=False,
                )
                if r is None or c is None:
                    # out of bounds (could be on a diagonal corner of the bbox)
                    continue

                val = da[..., r, c]
            else:
                val = da.sel(lat=row.lat, lon=row.lon, method="nearest")

            if np.any(np.isnan(val)) or np.any([np.all(val == v) for v in bad_vals]):
                continue
            else:
                # good_stations.append([row.name, row.lon, row.lat])
                good_idxs.append(row.Index)
        # to keep 'name' as column, but reset the former index to start at 0
        gdf_within = gdf_within.loc[good_idxs].reset_index(drop=True)
    return gdf_within


@lru_cache()
def read_station_llas(filename=None, to_geodataframe=False, force_download=True):
    """Read in the name, lat, lon, alt list of gps stations

    Assumes file is a space-separated with "name,lat,lon,alt" as columns
    """
    today = datetime.date.today().strftime("%Y%m%d")
    filename = filename or STATION_LLH_FILE.format(today=today)

    lla_path = os.path.join(GPS_DIR, filename)
    _remove_old_lists(lla_path)
    logger.debug("Searching %s for gps data" % filename)

    try:
        df = pd.read_csv(lla_path, sep=r"\s+", engine="c", header=None)
    except FileNotFoundError:
        logger.info("Downloading from %s to %s", STATION_LLH_URL, lla_path)
        download_station_locations(lla_path, STATION_LLH_URL)
        df = pd.read_csv(lla_path, sep=r"\s+", engine="c", header=None)

    df.columns = ["name", "lat", "lon", "alt"]
    if to_geodataframe:
        import geopandas as gpd

        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    else:
        return df


def _remove_old_lists(lla_path):
    today = datetime.date.today().strftime("%Y%m%d")
    gps_dir = os.path.split(lla_path)[0]
    station_list_files = glob(os.path.join(gps_dir, "station_*"))
    files_to_delete = [f for f in station_list_files if today not in f]
    for f in files_to_delete:
        logger.info("Removing old station list file: %s", f)
        os.remove(f)


@lru_cache()
def read_station_xyzs(filename=None):
    """Read in the name, X, Y, Z position of gps stations"""
    today = datetime.date.today().strftime("%Y%m%d")
    filename = filename or STATION_XYZ_FILE.format(today=today)
    _remove_old_lists(filename)
    logger.debug("Searching %s for gps data" % filename)
    try:
        df = pd.read_csv(
            filename,
            sep=r"\s+",
            engine="c",
            warn_bad_lines=True,
            error_bad_lines=False,
        )
    except FileNotFoundError:
        logger.warning("%s not found; downloading from %s", filename, STATION_XYZ_URL)
        download_station_locations(filename, STATION_XYZ_URL)
        df = pd.read_csv(
            filename,
            sep=r"\s+",
            engine="c",
            warn_bad_lines=True,
            error_bad_lines=False,
        )
    orig_cols = "Sta Lat(deg) Long(deg) Hgt(m) X(m) Y(m) Z(m) Dtbeg Dtend Dtmod NumSol StaOrigName"
    new_cols = "name lat lon alt X Y Z dtbeg dtend dtmod numsol origname"
    mapping = dict(zip(orig_cols.split(), new_cols.split()))
    return df.rename(columns=mapping)


def download_station_locations(filename, url):
    """Download either station LLH file or XYZ file from Nevada website
    url = [STATION_XYZ_URL or STATION_LLH_URL]
    """
    resp = requests.get(url)
    resp.raise_for_status()

    with open(filename, "w") as f:
        f.write(resp.text)


def download_station_data(station_name):
    station_name = station_name.upper()
    plate = _get_station_plate(station_name)
    # plate = "PA"
    url = GPS_BASE_URL.format(station=station_name, plate=plate)
    response = requests.get(url)
    response.raise_for_status()

    filename = os.path.join(GPS_DIR, GPS_FILE.format(station=station_name, plate=plate))
    logger.info(f"Saving {url} to {filename}")

    with open(filename, "w") as f:
        f.write(response.text)


def _get_station_plate(station_name):
    url = GPS_STATION_URL.format(station=station_name)
    response = requests.get(url)
    response.raise_for_status()

    # NOTE: This is not necessarily the only one!
    # CA GPS stations have PA and NA plate fixed... do i ever care about both?
    match = re.search(r"(?P<plate>[A-Z]{2}) Plate Fixed", response.text)
    if not match:
        raise ValueError("Could not find plate name on %s" % url)
    return match.groupdict()["plate"]


def station_lonlat(station_name):
    """Return the (lon, lat) in degrees of `station_name`"""
    df = read_station_llas()
    station_name = station_name.upper()
    if station_name not in df["name"].values:
        closest_names = difflib.get_close_matches(station_name, df["name"], n=5)
        raise ValueError(
            "No station named %s found. Closest: %s" % (station_name, closest_names)
        )
    name, lat, lon, alt = df[df["name"] == station_name].iloc[0]
    return lon, lat


def station_xyz(station_name):
    """Return the (X, Y, Z) in meters of `station_name`"""
    df = read_station_xyzs()
    station_name = station_name.upper()
    if station_name not in df["name"].values:
        closest_names = difflib.get_close_matches(station_name, df["name"], n=5)
        raise ValueError(
            "No station named %s found. Closest: %s" % (station_name, closest_names)
        )
    X, Y, Z = df.loc[df["name"] == station_name, ["X", "Y", "Z"]].iloc[0]
    return X, Y, Z


def station_rowcol(station_name, rsc_data=None, filename=None):
    """Find the row/columns of a station name within an image

    Image coordinates can be defined with `rsc_data` from .rsc file,
    or by a gdal-readable `filename`
    """
    if rsc_data is None and filename is None:
        raise ValueError("Need either rsc_data or filename to locate station")
    lon, lat = station_lonlat(station_name)
    return apertools.latlon.latlon_to_rowcol(
        lat, lon, rsc_data=rsc_data, filename=filename
    )


def station_distance(station_name1, station_name2):
    """Find distance (in meters) between two gps stations

    Args:
        station_name1 (str): name of first GPS station
        station_name2 (str): name of second GPS station

    Returns:
        float: distance (in meters)
    """
    lonlat1 = station_lonlat(station_name1)
    lonlat2 = station_lonlat(station_name2)
    return apertools.latlon.latlon_to_dist(lonlat1[::-1], lonlat2[::-1])


def station_std(station, to_cm=True, start_date=None, end_date=None):
    """Calculates the sum of east, north, and vertical stds of gps"""
    enu_df = load_station_enu(
        station, start_date=start_date, end_date=end_date, to_cm=to_cm
    )
    if enu_df.empty:
        logger.warning(f"{station} gps data returned an empty dataframe")
        return np.nan
    return np.sum(enu_df.std())


def load_gps_los(
    station_name=None,
    los_map_file=LOS_FILENAME,
    to_cm=True,
    zero_mean=True,
    zero_start=False,
    start_date=None,
    end_date=None,
    reference_station=None,
    enu_coeffs=None,
    force_download=False,
    coordinates="geo",
    geom_dir="geom_reference",
    days_smooth=0,
):
    """Load the GPS timeseries of a station name projected onto InSAR LOS

    Returns a DataFrame with index of date, one column for LOS measurement.

    This assumes that the los points AWAY from the satellite, towards the ground
    (subsidence is a positive LOS measurement, as it increases the LOS distance,
    and uplift is negative LOS)
    """
    if enu_coeffs is None:
        lon, lat = station_lonlat(station_name)
        enu_coeffs = apertools.los.find_enu_coeffs(
            lon,
            lat,
            los_map_file=los_map_file,
            coordinates=coordinates,
            geom_dir=geom_dir,
        )

    df_enu = load_station_enu(
        station_name,
        to_cm=to_cm,
        start_date=start_date,
        end_date=end_date,
        force_download=force_download,
    )
    enu_data = df_enu[["east", "north", "up"]].values.T
    los_gps_data = apertools.los.project_enu_to_los(enu_data, enu_coeffs=enu_coeffs)
    los_gps_data = los_gps_data.reshape(-1)

    if zero_start:
        logger.debug("Resetting GPS data start to 0")
        los_gps_data = los_gps_data - np.mean(los_gps_data[:100])
    elif zero_mean:
        logger.debug("Making GPS data 0 mean")
        los_gps_data = los_gps_data - np.mean(los_gps_data)

    if days_smooth:
        los_gps_data = moving_average(los_gps_data, days_smooth)

    df_los = pd.DataFrame(data=los_gps_data, index=df_enu.index, columns=["los"])
    if reference_station is not None:
        df_ref = load_gps_los(
            station_name=station_name,
            los_map_file=los_map_file,
            to_cm=to_cm,
            zero_mean=zero_mean,
            zero_start=zero_start,
            start_date=start_date,
            end_date=end_date,
            reference_station=None,
            enu_coeffs=enu_coeffs,
            force_download=force_download,
            days_smooth=days_smooth,
        )
        dfm = pd.merge(
            df_los,
            df_ref,
            how="inner",
            left_index=True,
            right_index=True,
            suffixes=("_target", "_ref"),
        )
        dfm.dropna(inplace=True)
        # Make a new columns with the reference subtracted off
        dfm["los"] = dfm["los_target"] - dfm["los_ref"]
        # Then drop the old original columns
        return dfm.drop(columns=["los_target", "los_ref"])

    return df_los


def moving_average(arr, window_size=7):
    """Takes a 1D array and returns the running average of same size"""
    if not window_size:
        return arr
    # return uniform_filter1d(arr, size=window_size, mode='nearest')
    return np.array(pd.Series(arr).rolling(window_size).mean())


def find_insar_ts(defo_filename, dset, station_name_list=[], window_size=1):
    """Get the insar timeseries closest to a list of GPS stations

    Returns the timeseries, and the datetimes of points for plotting

    Args:
        defo_filename
        station_name_list
        window_size (int): number of pixels in sqaure to average for insar timeseries
    """
    # slclist, deformation_stack = apertools.sario.load_deformation(full_path=defo_filename)
    # defo_img = apertools.latlon.load_deformation_img(full_path=defo_filename)

    insar_ts_list = []
    for station_name in station_name_list:
        lon, lat = station_lonlat(station_name)
        # row, col = defo_img.nearest_pixel(lat=lat, lon=lon)
        dem_rsc = apertools.sario.load_dem_from_h5(defo_filename)
        # TODO: use gdal/xarray here
        row, col = apertools.latlon.nearest_pixel(dem_rsc, lon=lon, lat=lat)
        insar_ts_list.append(
            get_stack_timeseries(
                defo_filename, row, col, station=station_name, window_size=window_size
            )
        )

    slclist = apertools.sario.load_slclist_from_h5(defo_filename, dset=dset)
    return slclist, insar_ts_list


def get_stack_timeseries(
    filename,
    row,
    col,
    stack_dset_name,
    station=None,
    window_size=1,
):
    with h5py.File(filename, "a") as f:
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


def _get_gps_insar_cols(df):
    gps_idxs = ["gps" in col for col in df.columns]
    insar_idxs = ["insar" in col for col in df.columns]
    gps_cols = df.columns[gps_idxs]
    insar_cols = df.columns[insar_idxs]
    return gps_idxs, gps_cols, insar_idxs, insar_cols


def _fit_line_to_dates(df):
    return np.array([linear_trend(df[col]).tail(1).squeeze() for col in df.columns])


def get_final_gps_insar_values(df, linear=True, as_df=False, velocity=True):
    # TODO: get rid of the "linear fit to unregularized"
    if linear:
        final_val_arr = _fit_line_to_dates(df)
    else:
        final_val_arr = df.tail(10).mean().values
    if velocity:
        full_dates = df.index
        days_spanning = (full_dates[-1] - full_dates[0]).days
        final_val_arr *= 10 * 365 / days_spanning  # Now in mm/year

    gps_idxs, gps_cols, insar_idxs, insar_cols = _get_gps_insar_cols(df)

    final_gps_vals = final_val_arr[gps_idxs]
    final_insar_vals = final_val_arr[insar_idxs]
    if not as_df:
        return gps_cols, insar_cols, final_gps_vals, final_insar_vals
    else:
        final_val_station_order = [s.split("_")[0] for s in gps_cols]
        return pd.DataFrame(
            index=final_val_station_order,
            data={"gps": final_gps_vals, "insar": final_insar_vals},
        )


def get_mean_correlations(defo_filename=None, dset=None, cor_filename="cor_stack.h5"):
    df_stations_available = get_stations_within_image(
        filename=defo_filename, dset=dset, mask_invalid=True
    )
    corrs = {}
    dem_rsc = apertools.sario.load_dem_from_h5(defo_filename)
    with h5py.File(cor_filename) as f:
        for tup in df_stations_available.itertuples():
            row, col = apertools.latlon.latlon_to_rowcol(tup.lat, tup.lon, dem_rsc)
            corrs[tup.name] = f["mean_stack"][row, col]
    return corrs


def save_station_points_kml(station_iter):
    for name, lat, lon, alt in station_iter:
        apertools.kml.create_kml(
            title=name,
            desc="GPS station location",
            lon_lat=(lon, lat),
            kml_out="%s.kml" % name,
            shape="point",
        )