from __future__ import division, print_function
import copy
from collections import Iterable
from numpy import sin, cos, sqrt, arctan2, radians
from xml.etree import ElementTree
import os
import numpy as np
import apertools.sario
import apertools.utils
from apertools.log import get_log

logger = get_log()


class LatlonImage(np.ndarray):
    """A class wrapping numpy ndarray with lat/lon info from DEM .rsc file data

    Has the ability to index values by [lat, lon] (as well as normal [row, col])
        lat, lon = 31.456, -102.54
        A[lat, lon]

    Can also slice:
        startlat, startlon = 31.456, -102.54
        endlat, endlon = 31.0, -102.0
        A[startlat:endlat, startlon:endlon]

    Caveat: remember that high latitudes are the top of the array,
    so if you want, e.g. N31.5 to N31.0, you need to say the higher latitude first.
    Pretend you are looking at a map with lat/lon, and just reading
    out the rows by their lats, columns by their lons


    In addition to the ndarray attributes, there are:

    Attributes:
        filename (str): Name of file where data was loaded from (if from file)
        rsc_file (str): filename of the dem.rsc file with info (see below)
        rsc_data (dict): lat/lon and file infomation about image
        dem_rsc_is_valid (bool): flag to indicate if the `rsc_data` applies to image.
            becomes invalid when, for example, 1D slicing occurs to get one row
    Keys from the rsc_data dict which are also attributes:
        width: number of cols in image (redundant with ncols)
        file_length: number of rows in image (redundant with nrows)
        x_first: location of first column
        y_first: location of first row
        x_step: size between rows in `x_unit`
        y_step: size between columns in `y_unit`
        x_unit (degrees, almost always)
        y_unit (degrees)
        z_offset (0)
        z_scale (1)
        projection (LL)
    """
    def __new__(cls, data=None, filename=None, rsc_file=None, rsc_data=None):
        """Can pass in either filenames to load, or 2D arrays/rsc_data dicts

        if __new__() returns an instance of cls, then the new instance's __init__()
        is then called
        Otherwise, the __init__ won't get called:
        https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#a-brief-python-primer-on-new-and-init

        reference: https://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        """
        # TODO: do we need to check that the rsc info matches the data?
        if data is None and filename is None:
            raise ValueError("Need data or filename")
        elif data is None and filename is not None:
            data = apertools.sario.load(filename, rsc_file=rsc_file)

        obj = np.asarray(data).view(cls)
        obj.filename = filename

        if rsc_file:
            obj.rsc_file = apertools.utils.fullpath(rsc_file)
        else:
            obj.rsc_file = apertools.sario.find_rsc_file(filename) if filename else None

        if rsc_data:
            obj.rsc_data = rsc_data
        elif obj.rsc_file:
            obj.rsc_data = apertools.sario.load(obj.rsc_file)
        else:
            obj.rsc_data = None

        # Set the flag to say whether a valid dem rsc exists/ is active
        # Also make each element in the rsc_data dict an object attr
        # e.g. : img.x_first
        if obj.rsc_data:
            obj.dem_rsc_is_valid = True
            for k, v in obj.rsc_data.items():
                setattr(obj, k, v)
        else:
            obj.dem_rsc_is_valid = False

        if obj.rsc_data is not None:
            if (obj.file_length, obj.width) != obj.shape[-2:]:
                raise ValueError("Shape %s does not equal rsc_data data (%s, %s)" %
                                 (obj.shape, obj.file_length, obj.width))

        # For things like keeping track of GPS points within image
        if not hasattr(obj, 'points'):
            obj.points = []

        return obj

    def __array_finalize__(self, obj):
        """Called when

        Reference:
        https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#the-role-of-array-finalize

        Deals with the view casting and new-from-template
        """
        if obj is None:
            return
        self.filename = getattr(obj, 'filename', None)
        self.rsc_file = getattr(obj, 'rsc_file', None)
        self.rsc_data = getattr(obj, 'rsc_data', None)
        self.dem_rsc_is_valid = getattr(obj, 'dem_rsc_is_valid', False)
        if self.rsc_data:
            for k, v in self.rsc_data.items():
                setattr(self, k, v)
        self.points = getattr(obj, 'points', None)

    def _disable_dem_rsc(self, sliced):
        """Some slice occurred which makes image data no longer apply"""
        sliced.dem_rsc_is_valid = False
        return sliced

    def __getitem__(self, items):
        """Runs on access/slicing: we want to adjust the rsc_data

        Will get the right starts and steps to pass to `crop_rsc_data`
        """
        # import ipdb
        # ipdb.set_trace()
        if contains_floats(items):
            items = self._convert_float_slice(items)
        sliced = super(LatlonImage, self).__getitem__(items)
        # __getitem__ called multiple times: only do extra on first
        if not isinstance(sliced, LatlonImage):
            return sliced
        # print('items')
        # print(items)
        # print('ndims', sliced.ndim, self.ndim)

        # print('here')
        # print(sliced, items)
        if not sliced.dem_rsc_is_valid or sliced.ndim < 2 or sliced.ndim > 3:
            return self._disable_dem_rsc(sliced)

        if self.ndim == 2:
            return self._handle_slice2(items, sliced)
        elif self.ndim == 3:
            return self._handle_slice3(items, sliced)

    def _handle_slice2(self, items, sliced):
        # We want this to stay 2D to have a valid dem.rsc
        if isinstance(items, slice):
            # This is a slice along rows, all cols
            return sliced
        elif isinstance(items, tuple) and all(isinstance(s, slice) for s in items):
            row_slice, col_slice = items
            # try:
            # except (TypeError, ValueError):
            # TypeError means they did A[100] or A[2:4], not A[2:4,:]
            # ValueError means something like an ndarray was used to index
            # We'll assume that this indexing will invalidate the rsc_data data
            # raise ValueError("Can only do 2D slices on %s" % self.__class__.__name__)
        else:
            return self._disable_dem_rsc(sliced)
        # do we need to check for Nones? the crop function seems to handle fine...
        # if row_slice == slice(None) or col_slice == slice(None):
        return self._handle_dem_slice(sliced, row_slice, col_slice)

    def _handle_slice3(self, items, sliced):
        if isinstance(items, int):
            # We asked for just one slice of stack, no need for further processing
            return sliced
        elif isinstance(items, slice):
            # We want a slice of the stack, still a 3d stack
            return sliced
        elif isinstance(items, tuple):
            # If we did something like A[:, :4, :4], crop the dem, still valid
            if len(items) == 3 and all(isinstance(s, slice) for s in items[1:]):
                _, row_slice, col_slice = items
                return self._handle_dem_slice(sliced, row_slice, col_slice)
            else:
                # Didn't pass 3 slices...  maybe a list of indexes
                return self._disable_dem_rsc(sliced)

    def _convert_float_slice(self, slice_items):
        """Convert lat/lon (float) slicing to row, cols

        Cases to handle:
            1. 2D image: A[lat, lon]
            2. 3D stack: A[:, lat, lon] (timeseries at pixel)
            3. slices within the lat, lon:
                A[lat, lon1:lon2]
        """
        # E.g. (slice(-101.1,100.0,.05), slice(30.0, 31.0, .05))
        if _is_float(slice_items):
            raise IndexError("Can't specify only 1 float for lat/lon indexing")
        # elif isinstance(slice_items, Iterable):
        if len(slice_items) == 3:
            if self.ndim == 3 or slice_items[0] is Ellipsis:
                # Something like [:, -101.1, 30.1] or [..., -101.1, 30.0]
                dslice = slice_items[0]
                lat, lon = slice_items[1:]
        elif len(slice_items) == 2 and self.ndim == 2:
            # Something like (-101.1, 30.0:30.2:.05) or (..., 101.1, 30.0)
            dslice = None
            lat, lon = slice_items
        else:
            raise IndexError("Invalid lat/lon slices for size %s LatlonImage: %s" %
                             (self.ndim, slice_items))

        if isinstance(lat, slice):
            # Use class step size if None given
            lat_start = lat.start or self.first_lat
            lat_stop = lat.stop or self.last_lat
            lat_step = lat.step or self.lat_step
            lat = np.arange(lat_start, lat_stop, lat_step)
        if isinstance(lon, slice):
            lon_start = lon.start or self.first_lon
            lon_stop = lon.stop or self.last_lon
            lon_step = lon.step or self.lon_step
            lon = np.arange(lon_start, lon_stop, lon_step)

        # print("Slicing lat = %s" % lat)
        # print("Slicing lon = %s" % lon)
        rows, cols = self.nearest_pixel(lon=lon, lat=lat)
        rows = rows[rows != np.array(None)]
        cols = cols[cols != np.array(None)]
        # print("Row: %s" % rows)
        # print("Col: %s" % cols)
        # TODO: do I care about converting lat slices to index slices?
        # for now assume continuous list if these are lists/arrays, we need them to be slices
        if isinstance(rows, Iterable):
            rows = slice(min(rows), max(rows) + 1)
        if isinstance(cols, Iterable):
            cols = slice(min(cols), max(cols) + 1)
        return (dslice, rows, cols) if dslice is not None else (rows, cols)

    def _handle_dem_slice(self, sliced, row_slice, col_slice):
        # print('sliced out shape', sliced.shape)
        # print('row_slice', row_slice, 'col slice', col_slice)
        # Note: we handled already returning slices of ndim < 2
        nrows, ncols = sliced.shape[-2:]

        row_start, row_step = row_slice.start, row_slice.step
        col_start, col_step = col_slice.start, col_slice.step
        new_rsc_data = self.crop_rsc_data(
            self.rsc_data,
            row_start,
            col_start,
            nrows,
            ncols,
            row_step,
            col_step,
        )

        sliced.rsc_data = new_rsc_data
        return sliced

    @staticmethod
    def crop_rsc_data(rsc_data, row_start, col_start, nrows, ncols, row_step=1, col_step=1):
        """Adjusts the old rsc_data for a cropped data

        Takes the 'file_length' and 'width' keys for a cropped data
        and adjusts for the smaller size with a new dict

        Returns:
            dict: copy of original rsc dict with items modified for crop

        Example:
        >>> im_test = np.arange(30).reshape((6, 5))
        >>> rsc_info = {'x_first': 1.0, 'y_first': 2.0, 'x_step': 0.1,\
'y_step': -0.2, 'file_length': 6,'width': 5}
        >>> im = LatlonImage(data=im_test, rsc_data=rsc_info)
        >>> out = im.crop_rsc_data(rsc_info, None, None, 2, 2)
        >>> print(out['width'], out['file_length'])
        2 2
        >>> out = im.crop_rsc_data(rsc_info, 1, 1, 2, 2)
        >>> print(out['x_first'], out['y_first'])
        1.1 1.8
        >>> out = im.crop_rsc_data(rsc_info, None, None, 2, 2, 2, 2)
        >>> print(out['x_step'], out['y_step'])
        0.2 -0.4
        >>> im2 = LatlonImage(data=im_test, rsc_data=None)
        >>> print(im.crop_rsc_data(None, 1, 4, 2, 5))
        None
        """
        if not rsc_data:
            return None

        # Adjust and Nones from the slice object
        row_start = row_start or 0
        col_start = col_start or 0
        row_step = row_step or 1
        col_step = col_step or 1

        rsc_copy = copy.copy(rsc_data)
        # Move forward the starting row/col from where it used to be
        rsc_copy['x_first'] = rsc_copy['x_first'] + rsc_copy['x_step'] * col_start
        rsc_copy['y_first'] = rsc_copy['y_first'] + rsc_copy['y_step'] * row_start

        rsc_copy['width'] = ncols
        rsc_copy['cols'] = ncols  # Also include the uavsar versions
        rsc_copy['file_length'] = nrows
        rsc_copy['rows'] = nrows
        # After moving start, now adjust step sizes
        rsc_copy['x_step'] *= col_step
        rsc_copy['y_step'] *= row_step
        return rsc_copy

    def take_looks(self, row_looks, col_looks):
        """Average array, and adjust rsc steps accordingly"""

        downlooked = apertools.utils.take_looks(self, row_looks, col_looks)

        row_start, col_start = None, None
        row_step, col_step = row_looks, col_looks
        nrows, ncols = downlooked.shape
        new_rsc_data = self.crop_rsc_data(
            self.rsc_data,
            row_start,
            col_start,
            nrows,
            ncols,
            row_step,
            col_step,
        )

        downlooked.rsc_data = new_rsc_data
        return downlooked

    @property
    def nrows(self):
        if len(self.shape) == 2:
            return self.shape[0]
        elif len(self.shape) == 3:
            return self.shape[1]
        else:
            raise ValueError("LatlonImage must be dim 2 or 3 to have nrows")

    @property
    def ncols(self):
        if len(self.shape) == 2:
            return self.shape[1]
        elif len(self.shape) == 3:
            return self.shape[2]
        else:
            raise ValueError("LatlonImage must be dim 2 or 3 to have ncols")

    def rowcol_to_latlon(self, row, col):
        return rowcol_to_latlon(row, col, self.rsc_data)

    @property
    def extent(self):
        if self.dem_rsc_is_valid:
            return grid_extent(**self.rsc_data)

    @property
    def top_left(self):
        """Returns the (lat, lon) of the top left corner"""
        if self.dem_rsc_is_valid:
            return self.x_first, self.y_first

    @property
    def last_lat(self):
        """The latitude of the last row of the image"""
        if self.dem_rsc_is_valid:
            return self.y_first + self.y_step * (self.shape[0] - 1)

    @property
    def last_lon(self):
        """The longitude of the last column of the image"""
        if self.dem_rsc_is_valid:
            return self.x_first + self.x_step * (self.shape[1] - 1)

    @property
    def first_lat(self):
        """alias to y_first, The latitude of the first row of the image"""
        if self.dem_rsc_is_valid:
            return self.y_first

    @property
    def first_lon(self):
        """alias to x_first, The longitude of the first column of the image"""
        if self.dem_rsc_is_valid:
            return self.x_first

    @property
    def lat_step(self):
        """The latitude increment for each pixel"""
        if self.dem_rsc_is_valid:
            return self.y_step

    @property
    def lon_step(self):
        """The longitude increment of each pixel"""
        if self.dem_rsc_is_valid:
            return self.x_step

    def nearest_pixel(self, lon=None, lat=None):
        """Find the nearest row, col to a given lat and/or lon

        Args:
            lon (ndarray[float]): single or array of lons
            lat (ndarray[float]): single or array of lat

        Returns:
            tuple[int, int]: If both given, a pixel (row, col) is returned
            If array passed for either lon or lat, array is returned
            Otherwise if only one, it is (None, col) or (row, None)
        """
        def _check_bounds(idx_arr, bound):
            int_idxs = idx_arr.round().astype(int)
            bad_idxs = np.logical_or(int_idxs < 0, int_idxs >= bound)
            if np.any(bad_idxs):
                # Need to check for single numbers, shape ()
                if int_idxs.shape:
                    # Replaces locations of bad_idxs with none
                    int_idxs = np.where(bad_idxs, None, int_idxs)
                else:
                    int_idxs = None
            return int_idxs

        out_row_col = [None, None]
        if lon is not None:
            col_idx_arr = (np.array(lon) - self.x_first) / self.x_step
            out_row_col[1] = _check_bounds(col_idx_arr, self.ncols)
        if lat is not None:
            row_idx_arr = (np.array(lat) - self.y_first) / self.y_step
            out_row_col[0] = _check_bounds(row_idx_arr, self.nrows)

        return tuple(out_row_col)

    def contains(self, lon_lat_point_list=None, lon_lat_point=None):
        if lon_lat_point is not None:
            lon, lat = lon_lat_point
            return grid_contains((lon, lat), **self.rsc_data)
            # Alternative:
            # Each of the tuple must contain an answer for the point to be contained
            # return all(num is not None for num in self.nearest_pixel(lon, lat))
        elif lon_lat_point_list is not None:
            return [grid_contains((lon, lat), **self.rsc_data) for lon, lat in lon_lat_point_list]

    def distance(self, row_col1, row_col2):
        """Find the distance in km between two points on the image

        Args:
            row_col1 (tuple[int, int]): starting (row, col)
            row_col2 (tuple[int, int]): ending (row, col)

        Returns:
            float: distance in km between two points on LatlonImage
        """
        latlon1 = self.rowcol_to_latlon(*row_col1)
        latlon2 = self.rowcol_to_latlon(*row_col2)
        return latlon_to_dist(latlon1, latlon2)

    def pixel_to_km(self, num_pixels):
        """Compute the length in km of segment `num_pixels` long

        Can also pass array to get multiple distances
        """
        num_pixels = np.array(num_pixels)
        # Use the center of the image as dummy anchor
        # (really only length should matter)
        nrows, ncols = self.shape
        midrow, midcol = nrows // 2, ncols // 2
        return self.distance((midrow, midcol), (midrow + num_pixels, midcol))

    def km_to_pixels(self, km):
        """Convert a km distance into number of pixels across"""
        deg_per_pixel = self.x_step  # assume x_step = y_step
        return km_to_pixels(km, deg_per_pixel)

    @property
    def km_per_pixel_sq(self):
        """Approximate area in one square pixel"""
        return self.pixel_to_km(1)**2


def load_deformation_img(igram_path=".",
                         n=1,
                         filename='deformation.h5',
                         dset=None,
                         full_path=None,
                         rsc_filename='dem.rsc'):
    """Loads mean of last n images of a deformation stack in LatlonImage
    Specify either a directory `igram_path` and `filename`, or `full_path` to file
    """
    igram_path, filename, full_path = apertools.sario.get_full_path(igram_path, filename, full_path)

    _, defo_stack = apertools.sario.load_deformation(igram_path=igram_path,
                                                     filename=filename,
                                                     full_path=full_path,
                                                     dset=dset,
                                                     n=n)
    if filename.endswith(".h5"):
        rsc_data = apertools.sario.load_dem_from_h5(h5file=full_path)
    else:
        rsc_data = apertools.sario.load(os.path.join(igram_path, rsc_filename))
    img = np.mean(defo_stack, axis=0) if n > 1 else defo_stack
    img = LatlonImage(data=img, rsc_data=rsc_data)
    return img


def rowcol_to_latlon(row, col, rsc_data):
    """ Takes the row, col of a pixel and finds its lat/lon

    Can also pass numpy arrays of row, col.
    row, col must match size

    Args:
        row (int or ndarray): row number
        col (int or ndarray): col number
        rsc_data (dict): data output from load_dem_rsc

    Returns:
        tuple[float, float]: lat, lon for the pixel

    Example:
        >>> rsc_data = {"x_first": 1.0, "y_first": 2.0, "x_step": 0.2, "y_step": -0.1}
        >>> rowcol_to_latlon(7, 3, rsc_data)
        (1.4, 1.4)
    """
    start_lon = rsc_data["x_first"]
    start_lat = rsc_data["y_first"]
    lon_step, lat_step = rsc_data["x_step"], rsc_data["y_step"]
    lat = start_lat + (row - 1) * lat_step
    lon = start_lon + (col - 1) * lon_step

    return lat, lon


def latlon_to_rowcol(lat, lon, rsc_data):
    """Takes latitude, longitude and finds pixel location.

    Inverse of rowcol_to_latlon function

    Args:
        lat (float): latitude
        lon (float): longitude
        rsc_data (dict): data output from load_dem_rsc

    Returns:
        tuple[int, int]: row, col for the pixel

    Example:
        >>> rsc_data = {"x_first": 1.0, "y_first": 2.0, "x_step": 0.2, "y_step": -0.1}
        >>> latlon_to_rowcol(1.4, 1.4, rsc_data)
        (6, 2)
        >>> latlon_to_rowcol(2.0, 1.0, rsc_data)
        (0, 0)
    """
    start_lon = rsc_data["x_first"]
    start_lat = rsc_data["y_first"]
    lon_step, lat_step = rsc_data["x_step"], rsc_data["y_step"]
    row = (lat - start_lat) / lat_step
    col = (lon - start_lon) / lon_step
    return int(round(row)), int(round(col))


def latlon_to_dist(lat_lon_start, lat_lon_end, R=6378):
    """Find the distance between two lat/lon points on Earth

    Uses the haversine formula: https://en.wikipedia.org/wiki/Haversine_formula
    so it does not account for the ellopsoidal Earth shape. Will be with about
    0.5-1% of the correct value.

    Notes: lats and lons are in degrees, and the values used for R Earth
    (6373 km) are optimized for locations around 39 degrees from the equator

    Reference: https://andrew.hedges.name/experiments/haversine/

    Args:
        lat_lon_start (tuple[int, int]): (lat, lon) in degrees of start
        lat_lon_end (tuple[int, int]): (lat, lon) in degrees of end
        R (float): Radius of earth

    Returns:
        float: distance between two points in km

    Examples:
        >>> round(latlon_to_dist((38.8, -77.0), (38.9, -77.1)), 1)
        14.1
    """
    lat1, lon1 = lat_lon_start
    lat2, lon2 = lat_lon_end
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = (sin(dlat / 2)**2) + (cos(lat1) * cos(lat2) * sin(dlon / 2)**2)
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))
    return R * c


def km_to_deg(km, R=6378):
    """Find the degrees separation for distance km

    Assumes distance along great circle arc

    Args:
        km (float): distance in degrees
        R (float): default 6378, Radius of Earth in km

    Returns:
        float: distance in degrees
    """
    return 360 * km / (2 * np.pi * R)


# Alias to match latlon_to_dist
dist_to_deg = km_to_deg


def km_to_pixels(km, step, R=6378):
    """Convert km distance to pixel size

    Note: This assumes x_step, y_step are equal, and
    calculates the distance in a vertical direction
    (which is more pixels than in the diagonal direction)

    Args:
        km (float): distance in degrees
        step (float): number of degrees per pixel step
        R (float): default 6378, Radius of Earth in km

    Returns:
        float: distance in number of pixels
    """
    degrees = km_to_deg(km, R)
    return degrees / step


dist_to_pixel = km_to_pixels


def grid(rows=None,
         cols=None,
         y_step=None,
         x_step=None,
         y_first=None,
         x_first=None,
         width=None,
         file_length=None,
         sparse=False,
         **kwargs):
    """Takes sizes and spacing info, creates a grid of values

    Args:
        rows (int): number of rows
        cols (int): number of cols
        y_step (float): spacing between rows
        x_step (float): spacing between cols
        y_first (float): starting location of first row at top
        x_first (float): starting location of first col on left
        sparse (bool): Optional (default False). Passed through to
            np.meshgrid to optionally conserve memory

    Returns:
        tuple[ndarray, ndarray]: the XX, YY grids of longitudes and lats

    Examples:
    >>> test_grid_data = {'cols': 2, 'rows': 3, 'x_first': -155.0, 'x_step': 0.01,\
'y_first': 19.5, 'y_step': -0.2}
    >>> lons, lats = grid(**test_grid_data)
    >>> np.set_printoptions(legacy="1.13")
    >>> print(lons)
    [[-155.   -154.99]
     [-155.   -154.99]
     [-155.   -154.99]]
    >>> print(lats)
    [[ 19.5  19.5]
     [ 19.3  19.3]
     [ 19.1  19.1]]
    """
    rows = rows or file_length
    cols = cols or width
    x = np.linspace(x_first, x_first + (cols - 1) * x_step, cols).reshape((1, cols))
    y = np.linspace(y_first, y_first + (rows - 1) * y_step, rows).reshape((rows, 1))
    return np.meshgrid(x, y, sparse=sparse)


def grid_to_rsc(lons, lats):
    """Reverses the `grid` function to get an rsc dict

    Takes the meshgrid output `lons` and `lats` and calculates the rsc data
    """
    assert lons.shape == lats.shape
    rows, cols = lons.shape
    file_length, width = rows, cols

    # Check that the northern most latitude is on the top row for lats
    if lats[0][0] < lats[-1][0]:
        raise ValueError("Need top row of lats to be the northern most lats")

    y_first = lats[0][0]
    y_step = lats[1][0] - lats[0][0]

    x_first = lons[0][0]
    x_step = lons[0][1] - lons[0][0]

    return dict(
        x_first=x_first,
        y_first=y_first,
        rows=rows,
        cols=cols,
        width=width,
        file_length=file_length,
        x_step=x_step,
        y_step=y_step,
    )


def grid_extent(rows=None,
                cols=None,
                y_step=None,
                x_step=None,
                y_first=None,
                x_first=None,
                file_length=None,
                width=None,
                **kwargs):
    """Takes sizes and spacing from .rsc info, finds boundaries

    Used for `matplotlib.pyplot.imshow` keyword arg `extent`:
    extent : scalars (left, right, bottom, top)

    Args:
        rows (int): number of rows
        cols (int): number of cols
        y_step (float): spacing between rows
        x_step (float): spacing between cols
        y_first (float): starting location of first row at top
        x_first (float): starting location of first col on left
        file_length (int): alias for number of rows (used in dem.rsc)
            Not needed if `rows` is supplied
        width (int): alias for number of cols (used in dem.rsc)
            Not needed if `cols` is supplied

    Returns:
        tuple[float]: the boundaries of the latlon grid in order:
        (lon_left,lon_right,lat_bottom,lat_top)

    Examples:
    >>> test_grid_data = {'cols': 2, 'rows': 3, 'x_first': -155.0, 'x_step': 0.01,\
'y_first': 19.5, 'y_step': -0.2}
    >>> print(grid_extent(**test_grid_data))
    (-155.0, -154.99, 19.1, 19.5)
    """
    rows = rows or file_length
    cols = cols or width
    return (x_first, x_first + x_step * (cols - 1), y_first + y_step * (rows - 1), y_first)


def grid_corners(**kwargs):
    """Takes sizes and spacing from .rsc info, finds corner points in (x, y) form

    Returns:
        list[tuple[float]]: the corners of the latlon grid in order:
        (top right, top left, bottom left, bottom right)
    """
    left, right, bot, top = grid_extent(**kwargs)
    return [(right, top), (left, top), (left, bot), (right, bot)]


def grid_midpoint(**kwargs):
    """Takes sizes and spacing from .rsc info, finds midpoint in (x, y) form

    Returns:
        tuple[float]: midpoint of the latlon grid
    """
    left, right, bot, top = grid_extent(**kwargs)
    return (left + right) / 2, (top + bot) / 2


def grid_size(**kwargs):
    """Takes rsc_data and gives width and height of box in km

    Returns:
        tupls[float, float]: width, height in km
    """
    left, right, bot, top = grid_extent(**kwargs)
    width = latlon_to_dist((top, left), (top, right))
    height = latlon_to_dist((top, left), (bot, right))
    return width, height


def grid_bounds(**kwargs):
    """Same to grid_extent (takes .rsc info) , but in the order (left, bottom, right, top)"""
    left, right, bot, top = grid_extent(**kwargs)
    return left, bot, right, top


def grid_width_height(**kwargs):
    """Finds the width and height in deg of the latlon grid from .rsc info"""
    left, right, bot, top = grid_extent(**kwargs)
    return (right - left, top - bot)


def grid_contains(point, **kwargs):
    """Returns true if point (x, y) or (lon, lat) is within the grid. Takes .rsc info"""
    point_x, point_y = point
    left, right, bot, top = grid_extent(**kwargs)
    return (left < point_x < right) and (bot < point_y < top)


def intersects1d(low1, high1, low2, high2):
    """Checks if two line segments intersect

    Example:
    >>> low1, high1 = [1, 5]
    >>> low2, high2 = [4, 6]
    >>> print(intersects1d(low1, high1, low2, high2))
    True
    >>> low2 = 5.5
    >>> print(intersects1d(low1, high1, low2, high2))
    False
    >>> high1 = 7
    >>> print(intersects1d(low1, high1, low2, high2))
    True
    """
    # Is this easier?
    # return not (high2 <= low1 or high2 <= low1)
    return high1 >= low2 and high2 >= low1


def intersects(box1, box2):
    """Returns true if box1 intersects box2

    box = (left, right, bot, top), same as matplotlib `extent` format

    Example:
    >>> box1 = (-105.1, -102.2, 31.4, 33.4)
    >>> box2 = (-103.4, -102.7, 30.9, 31.8)
    >>> print(intersects(box1, box2))
    True
    >>> box2 = (-103.4, -102.7, 30.9, 31.0)
    >>> print(intersects(box2, box1))
    False
    """
    return intersect_area(box1, box2) > 0


def box_area(box):
    """Returns area of box from format (left, right, bot, top)
    Example:
    >>> box1 = (-1, 1, -1, 1)
    >>> print(box_area(box1))
    4
    """
    left, right, bot, top = box
    dx = np.clip(right - left, 0, None)
    dy = np.clip(top - bot, 0, None)
    return dx * dy


def _check_valid_box(box):
    left, right, bot, top = box
    if (left > right) or (bot > top):
        raise ValueError("Box %s must be in form (left, right, bot, top)" % str(box))


def intersect_area(box1, box2):
    """Returns area of overlap of two rectangles

    box = (left, right, bot, top), same as matplotlib `extent` format
    Example:
    >>> box1 = (-1, 1, -1, 1)
    >>> box2 = (-1, 1, 0, 2)
    >>> print(intersect_area(box1, box2))
    2
    >>> box2 = (0, 2, -1, 1)
    >>> print(intersect_area(box1, box2))
    2
    >>> box2 = (4, 6, -1, 1)
    >>> print(intersect_area(box1, box2))
    0
    """
    _check_valid_box(box1), _check_valid_box(box2)
    left1, right1, bot1, top1 = box1
    left2, right2, bot2, top2 = box2
    intersect_box = (max(left1, left2), min(right1, right2), max(bot1, bot2), min(top1, top2))
    return box_area(intersect_box)


def union_area(box1, box2):
    """Returns area of union of two rectangles, which is A1 + A2 - intersection

    box = (left, right, bot, top), same as matplotlib `extent` format
    >>> box1 = (-1, 1, -1, 1)
    >>> box2 = (-1, 1, 0, 2)
    >>> print(union_area(box1, box2))
    6
    >>> print(union_area(box1, box1) == box_area(box1))
    True
    """
    _check_valid_box(box1), _check_valid_box(box2)
    A1 = box_area(box1)
    A2 = box_area(box2)
    return A1 + A2 - intersect_area(box1, box2)


def intersection_over_union(box1, box2):
    """Returns the IoU critera for pct of overlap area

    box = (left, right, bot, top), same as matplotlib `extent` format
    >>> box1 = (0, 1, 0, 1)
    >>> box2 = (0, 2, 0, 2)
    >>> print(intersection_over_union(box1, box2))
    0.25
    >>> print(intersection_over_union(box1, box1))
    1.0
    """
    ua = union_area(box1, box2)
    if ua == 0:
        return 0
    else:
        return intersect_area(box1, box2) / ua


def intersection_corners(dem1, dem2):
    """
    Returns:
        tuple[float]: the boundaries of the intersection box of the 2 areas in order:
        (lon_left,lon_right,lat_bottom,lat_top)
    """
    def _max_min(a, b):
        """The max of two iterable mins"""
        return max(min(a), min(b))

    def _least_common(a, b):
        """The min of two iterable maxes"""
        return min(max(a), max(b))

    corners1 = grid_corners(**dem1)
    corners2 = grid_corners(**dem2)
    lons1, lats1 = zip(*corners1)
    lons2, lats2 = zip(*corners2)
    left = _max_min(lons1, lons2)
    right = _least_common(lons1, lons2)
    bottom = _max_min(lats1, lats2)
    top = _least_common(lats1, lats2)
    return left, right, bottom, top


def sort_by_lat(latlon_img_list):
    """Sorts a list of LatlonImages by latitude, north to south"""
    return sorted(latlon_img_list, key=lambda img: img.y_first, reverse=True)


def sort_by_lon(latlon_img_list):
    """Sorts a list of LatlonImages by longitude, west to east"""
    return sorted(latlon_img_list, key=lambda img: img.x_first)


def find_img_intersections(image1, image2):
    """Takes two LatlonImages, finds which pixels mark their intersection
    """
    if not intersects(image1.extent, image2.extent):
        return (None, None)

    # TODO: i'm messing up the order between bottom right/ left overlap, e.g.
    im1_lat, im2_lat = sort_by_lat([image1, image2])
    im1_lon, im2_lon = sort_by_lon([image1, image2])

    lat_tup = im1_lat.nearest_pixel(lat=im2_lat.y_first)
    lon_tup = im1_lon.nearest_pixel(lon=im2_lon.x_first)

    # Now combine by ignoring the Nones for each that don't matter
    return (lat_tup[0], lon_tup[1])


def find_total_pixels(image_list):
    """Get the total number of rows and columns for overlapping images
    """
    # TODO: + 1 needed?
    if any(not img.dem_rsc_is_valid for img in image_list):
        raise ValueError("All images must have rsc_data provided")
    elif any(img.x_step != image_list[0].x_step for img in image_list):
        raise ValueError("All images must have same x_step in rsc_data")
    elif any(img.y_step != image_list[0].y_step for img in image_list):
        raise ValueError("All images must have same y_step in rsc_data")

    images_sorted = sort_by_lat(image_list)
    im_first = images_sorted[0]
    im_last = images_sorted[-1]
    row_increments = int(round((im_last.last_lat - im_first.y_first) / im_first.y_step))

    images_sorted = sort_by_lon(image_list)
    im_first = images_sorted[0]
    im_last = images_sorted[-1]
    col_increments = int(round((im_last.last_lon - im_first.x_first) / im_first.x_step))
    # Add 1 to count the starting row
    return (1 + row_increments, 1 + col_increments)


def map_overlay_coords(kml_file=None, etree=None):
    if not os.path.exists(kml_file):
        return None
    # Use the cache doesn't exist, parse xml and save it
    if kml_file:
        etree = ElementTree.parse(kml_file)
    if not etree:
        raise ValueError("Need xml_file or etree")

    root = etree.getroot()
    # point_str looks like:
    # <coordinates>-102.552971,31.482372 -105.191353,31.887299...
    point_str = list(elem.text for elem in root.iter('coordinates'))[0]
    return [(float(lon), float(lat)) for lon, lat in [p.split(',') for p in point_str.split()]]


def _is_float(a):
    return isinstance(a, float)


def contains_floats(slice_items):
    """Checks possible slice item values for floats"""
    if _is_float(slice_items):
        return True
    elif isinstance(slice_items, Iterable):
        for item in slice_items:
            if _is_float(item):
                return True
            elif isinstance(item, slice):
                if _is_float(item.start) or _is_float(item.stop):
                    return True
    return False


def load_cropped_masked_deformation(path=".",
                                    filename="deformation.h5",
                                    dset=None,
                                    full_path=None,
                                    rsc_name="dem.rsc",
                                    row_start=0,
                                    row_end=None,
                                    col_start=0,
                                    col_end=None,
                                    n=None,
                                    perform_mask=True):
    """Returns stack_ll, 3D LatlonImage, and stack_mask, used to mask the data"""
    path, filename, full_path = apertools.sario.get_full_path(path, filename, full_path)
    geo_date_list, deformation = apertools.sario.load_deformation(
        igram_path=path,
        filename=filename,
        full_path=full_path,
        n=n,
    )

    if geo_date_list is None or deformation is None:
        return

    rsc_data = apertools.sario.load(os.path.join(path, rsc_name))
    stack_mask = apertools.sario.load_mask(geo_date_list=geo_date_list,
                                           perform_mask=perform_mask,
                                           directory=path,
                                           dset=dset)

    stack_ll = LatlonImage(data=deformation, rsc_data=rsc_data)
    stack_ll[:, stack_mask] = np.nan

    stack_ll = stack_ll[..., row_start:row_end, col_start:col_end]
    return stack_ll, stack_mask


# TODO: should this just roll into latlon_to_rowcol?
def nearest_pixel(rsc_data, lon=None, lat=None, ncols=np.inf, nrows=np.inf):
    """Find the nearest row, col to a given lat and/or lon within rsc_data

    Args:
        lon (ndarray[float]): single or array of lons
        lat (ndarray[float]): single or array of lat

    Returns:
        tuple[int, int]: If both given, a pixel (row, col) is returned
        If array passed for either lon or lat, array is returned
        Otherwise if only one, it is (None, col) or (row, None)
    """
    def _check_bounds(idx_arr, bound):
        int_idxs = idx_arr.round().astype(int)
        bad_idxs = np.logical_or(int_idxs < 0, int_idxs >= bound)
        if np.any(bad_idxs):
            # Need to check for single numbers, shape ()
            if int_idxs.shape:
                # Replaces locations of bad_idxs with none
                int_idxs = np.where(bad_idxs, None, int_idxs)
            else:
                int_idxs = None
        return int_idxs

    out_row_col = [None, None]

    if lon is not None:
        out_row_col[1] = _check_bounds(nearest_col(rsc_data, lon), ncols)
    if lat is not None:
        out_row_col[0] = _check_bounds(nearest_row(rsc_data, lat), nrows)

    return tuple(out_row_col)


def nearest_row(rsc_data, lat):
    """Find the nearest row to a given lat within rsc_data (no OOB checking)"""
    y_first, y_step = rsc_data['y_first'], rsc_data['y_step']
    return ((np.array(lat) - y_first) / y_step).round().astype(int)


def nearest_col(rsc_data, lon):
    """Find the nearest col to a given lon within rsc_data (no OOB checking)"""
    x_first, x_step = rsc_data['x_first'], rsc_data['x_step']
    return ((np.array(lon) - x_first) / x_step).round().astype(int)
