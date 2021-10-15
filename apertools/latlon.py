from xml.etree import ElementTree
import os
import numpy as np

# TODO: lots of stuff here can be done with pyproj.Geod
# https://pyproj4.github.io/pyproj/stable/api/geod.html
from pyproj import Geod

from apertools.log import get_log

WGS84 = Geod(ellps="WGS84")

logger = get_log()


def rowcol_to_latlon(row, col, rsc_data=None, filename=None):
    """Takes the row, col of a pixel and finds its lat/lon

    Can also pass numpy arrays of row, col.
    row, col must match size

    Args:
        row (int or ndarray): row number
        col (int or ndarray): col number
        rsc_data (dict): data output from load_dem_rsc
        filename (str): gdal-readable file with geographic coordinates

    Returns:
        tuple[float, float]: lat, lon for the pixel

    Example:
        >>> rsc_data = {"x_first": 1.0, "y_first": 2.0, "x_step": 0.2, "y_step": -0.1}
        >>> rowcol_to_latlon(7, 3, rsc_data)
        (1.4, 1.4)
    """
    if rsc_data is not None:
        start_lon = rsc_data["x_first"]
        start_lat = rsc_data["y_first"]
        lon_step, lat_step = rsc_data["x_step"], rsc_data["y_step"]
        lat = start_lat + (row - 1) * lat_step
        lon = start_lon + (col - 1) * lon_step
    else:
        import rasterio as rio

        with rio.open(filename) as src:
            lon, lat = src.xy(row, col)

    return lat, lon


def latlon_to_rowcol(lat, lon, rsc_data=None, filename=None):
    """Takes latitude, longitude and finds pixel location.

    Inverse of rowcol_to_latlon function

    Args:
        lat (float): latitude
        lon (float): longitude
        rsc_data (dict): data output from load_dem_rsc
        filename (str): gdal-readable file with geographic coordinates

    Returns:
        tuple[int, int]: row, col for the pixel

    Example:
        >>> rsc_data = {"x_first": 1.0, "y_first": 2.0, "x_step": 0.2, "y_step": -0.1}
        >>> latlon_to_rowcol(1.4, 1.4, rsc_data)
        (6, 2)
        >>> latlon_to_rowcol(2.0, 1.0, rsc_data)
        (0, 0)
    """
    if rsc_data is None and filename is None:
        raise ValueError("Need either rsc_data or filename to locate station")

    if rsc_data is not None:
        start_lon = rsc_data["x_first"]
        start_lat = rsc_data["y_first"]
        lon_step, lat_step = rsc_data["x_step"], rsc_data["y_step"]
        row = (lat - start_lat) / lat_step
        col = (lon - start_lon) / lon_step

        row, col = int(round(row)), int(round(col))
    else:
        import rasterio as rio

        with rio.open(filename) as src:
            row, col = src.index(lon, lat)
    return row, col


def latlon_to_dist(lat_lon_start, lat_lon_end):
    """Find the distance between two lat/lon points on Earth [in meters]

    lats and lons are in degrees, WGS84 ellipsoid is used
    wrapper around pyproj.Geod for older compatibility

    Args:
        lat_lon_start (tuple[int, int]): (lat, lon) in degrees of start
        lat_lon_end (tuple[int, int]): (lat, lon) in degrees of end

    Returns:
        float: distance between two points in meters

    Examples:
        >>> round(latlon_to_dist((38.8, -77.0), (38.9, -77.1)))
        14092
    """
    lat1, lon1 = lat_lon_start
    lat2, lon2 = lat_lon_end
    return WGS84.line_length((lon1, lon2), (lat1, lat2))


def pixel_spacing(
    y_step=None,
    x_step=None,
    y_first=None,
    x_first=None,
    img_xr=None,
    lat_arr=None,
    lon_arr=None,
    **kwargs,
):
    """Return the (x_spacing, y_spacing) pixel spacing in meters, given degrees"""
    if img_xr is not None:
        lon_arr = img_xr.lon
        lat_arr = img_xr.lat
    if lat_arr is not None and lon_arr is not None:
        x_first, y_first = lon_arr[0], lat_arr[0]
        x_step = lon_arr[1] - lon_arr[0]
        y_step = lat_arr[1] - lat_arr[0]

    start_latlon = (y_first, x_first)
    end_x = (y_first, x_first + x_step)
    x_spacing = latlon_to_dist(start_latlon, end_x)

    end_y = (y_first + y_step, x_first)
    y_spacing = latlon_to_dist(start_latlon, end_y)
    return x_spacing, y_spacing


def get_res_reproject(img):
    """TODO: why is this 50 m off from the geoid distance?"""
    test = img[:3, :3]
    test.rio.write_crs("EPSG:4326", inplace=True)
    test_xy = test.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.reproject(
        "EPSG:3857"
    )
    x_res, y_res = test_xy.x.diff("x")[0].item(), -test_xy.y.diff("y")[0].item()
    return x_res, y_res


def km_to_deg(km, R=6378):
    """Find the degrees separation for distance km

    Assumes distance along great circle arc

    Args:
        km (float, ndarray[float]): distance in degrees
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


def grid(
    rows=None,
    cols=None,
    y_step=None,
    x_step=None,
    y_first=None,
    x_first=None,
    width=None,
    file_length=None,
    sparse=False,
    fname=None,
    **kwargs,
):
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
    if fname is not None:
        transform_data = get_tif_transform(fname)
        return grid(sparse=sparse, **transform_data)

    rows = rows or file_length
    cols = cols or width
    x = np.linspace(x_first, x_first + (cols - 1) * x_step, cols).reshape((1, cols))
    y = np.linspace(y_first, y_first + (rows - 1) * y_step, rows).reshape((rows, 1))
    return np.meshgrid(x, y, sparse=sparse)


def get_tif_transform(fname):
    """Make a dict containing similar info to .rsc file from GDAL-readable file"""
    import rasterio as rio

    with rio.open(fname) as src:
        x_step, _, x_first, _, y_step, y_first, _, _, _ = tuple(src.transform)
        rows, cols = src.shape
        return dict(
            rows=rows,
            cols=cols,
            x_step=x_step,
            x_first=x_first,
            y_step=y_step,
            y_first=y_first,
        )


def grid_to_rsc(lons, lats, sparse=False):
    """Reverses the `grid` function to get an rsc dict

    Takes the meshgrid output `lons` and `lats` and calculates the rsc data
    """
    if sparse is False:
        assert lons.shape == lats.shape
        rows, cols = lons.shape
        y_first = lats[0][0]
        y_step = lats[1][0] - lats[0][0]

        x_first = lons[0][0]
        x_step = lons[0][1] - lons[0][0]
    else:
        lats, lons = lats.reshape(-1), lons.reshape(-1)
        rows, cols = len(lats), len(lons)
        y_first = lats[0]
        y_step = lats[1] - lats[0]
        x_first = lons[0]
        x_step = lons[1] - lons[0]

    file_length, width = rows, cols

    # Check that the northern most latitude is on the top row for lats
    # if lats[0][0] < lats[-1][0]:
    # raise ValueError("Need top row of lats to be the northern most lats")

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


# should this be in another file?
def get_latlon_arrs(h5_filename=None, dem_rsc_file=None, gdal_file=None, bbox=None):
    import apertools.sario

    if h5_filename is not None:
        lon_arr, lat_arr = grid(
            **apertools.sario.load_dem_from_h5(h5_filename), sparse=True
        )
    elif dem_rsc_file is not None:
        lon_arr, lat_arr = grid(**apertools.sario.load(dem_rsc_file), sparse=True)
    elif gdal_file is not None:
        import rasterio as rio

        with rio.open(gdal_file) as src:
            rows, cols = src.shape
            max_len = max(rows, cols)
            lon_list, lat_list = src.xy(np.arange(max_len), np.arange(max_len))
            lon_arr = np.arange(lon_list[:cols])
            lat_arr = np.arange(lat_list[:rows])

    lon_arr, lat_arr = lon_arr.reshape(-1), lat_arr.reshape(-1)
    return lon_arr.reshape(-1), lat_arr.reshape(-1)


def from_grid(lons, lats, sparse=False):
    """Alias for grid_to_rsc"""
    return grid_to_rsc(lons, lats, sparse=sparse)


def grid_extent(
    rows=None,
    cols=None,
    y_step=None,
    x_step=None,
    y_first=None,
    x_first=None,
    file_length=None,
    width=None,
    **kwargs,
):
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
    return (
        x_first,
        x_first + x_step * (cols - 1),
        y_first + y_step * (rows - 1),
        y_first,
    )


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


def window_rowcol(lon_arr, lat_arr, bbox=None):
    """Get the row bounds and col bounds of a box in lat/lon arrays

    Returns:
        (row_top, row_bot), (col_left, col_right)
    """
    if bbox is None or len(bbox) == 0:
        return (0, len(lat_arr)), (0, len(lon_arr))

    left, bot, right, top = bbox
    lat_step = np.diff(lat_arr)[0]
    lon_step = np.diff(lon_arr)[0]
    row_top = np.clip(int(round(((top - lat_arr[0]) / lat_step))), 0, len(lat_arr))
    row_bot = np.clip(int(round(((bot - lat_arr[0]) / lat_step))), 0, len(lat_arr))
    col_left = np.clip(int(round(((left - lon_arr[0]) / lon_step))), 0, len(lon_arr))
    col_right = np.clip(int(round(((right - lon_arr[0]) / lon_step))), 0, len(lon_arr))
    # lat_arr = lon_arr[row_top:row_bot]
    # lon_arr = lon_arr[col_left:col_right]
    return (row_top, row_bot), (col_left, col_right)


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
    intersect_box = (
        max(left1, left2),
        min(right1, right2),
        max(bot1, bot2),
        min(top1, top2),
    )
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
    point_str = list(elem.text for elem in root.iter("coordinates"))[0]
    return [
        (float(lon), float(lat))
        for lon, lat in [p.split(",") for p in point_str.split()]
    ]


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
    y_first, y_step = rsc_data["y_first"], rsc_data["y_step"]
    return ((np.array(lat) - y_first) / y_step).round().astype(int)


def nearest_col(rsc_data, lon):
    """Find the nearest col to a given lon within rsc_data (no OOB checking)"""
    x_first, x_step = rsc_data["x_first"], rsc_data["x_step"]
    return ((np.array(lon) - x_first) / x_step).round().astype(int)


def bbox_xr(dataset):
    """Get the lon/lat bounding box (minx, miny, maxx, maxy) from an xarray dataset"""
    if "lat" in dataset.coords:
        y = dataset["lat"]
    elif "y" in dataset.coords:
        y = dataset["y"]
    else:
        raise ValueError("dataset {} must contain 'lat' or 'y'".format(dataset))
    if "lon" in dataset.coords:
        x = dataset["lon"]
    elif "x" in dataset.coords:
        x = dataset["lon"]
    else:
        raise ValueError("dataset {} must contain 'lon' or 'x'".format(dataset))
    return x.min(), y.min(), x.max(), y.max()


# ###### ISCE/ Radar coordinate functions #######


def bbox_from_latlon_arrs(lon_arr, lat_arr):
    """Generate the (left, bot, right, top) latitudes/longitudes from radar geometry images"""
    left, right = lon_arr.min(), lon_arr.max()
    bot, top = lat_arr.min(), lat_arr.max()
    return left, bot, right, top


def latlon_to_rowcol_rdr(
    lat, lon, lat_arr=None, lon_arr=None, geom_dir=None, warn_oob=True, looks=None
):
    """Find the row/col in radar coordinates (azimuth/range index) for a lat/lon point

    Args:
        lat (float): latitude of point of interest
        lon (float): longitude of point of interest
        lat_arr (ndarray, optional): lat geometry array for radar coordinates.
        lon_arr (ndarray, optional): lon geometry array for radar coordinates.
        geom_dir (str, optional): directory containing the lat/lon arrays to load.

    Raises:
        ValueError: If none of lat_arr/lon_arr/geom_dir are provided

    Returns:
        (row, col) for the (az, range) position of point closest to lat/lon
    """
    import apertools.sario

    if lat_arr is None or lon_arr is None:
        if geom_dir is None:
            raise ValueError("need either lat/lon geomaetry arrays or geom_dir")
        lat_arr, lon_arr = apertools.sario.load_rdr_latlon(geom_dir=geom_dir)
    # Find the minimum distance location (and convert from linear index to row/col)
    dlat = np.abs(lat_arr - lat)
    dlon = np.abs(lon_arr - lon)
    # Get the strip from `lat_arr`/`lon_arr` within `buf` of the desired lat/lon
    min_diff = np.min(np.abs(np.diff(lat_arr.ravel())))
    buf = max(1e-3, 1.5 * min_diff)
    matching_lat = dlat < buf
    matching_lon = dlon < buf
    found_area = np.logical_and(matching_lat, matching_lon)
    # Take the centroid of the matching area (two strips make some quad shape)
    rows, cols = np.where(found_area)

    if not rows.size or not cols.size:
        if warn_oob:
            logger.warning(f"{lat = }, {lon = } is outside latitude array bounds")
        return None, None
    row, col = round(np.mean(rows)), round(np.mean(cols))
    # If we only had the looked geometry files, multiply for the SLC version:
    if looks is not None:
        row_looks, col_looks = looks
        row *= row_looks
        col *= col_looks
    return row, col


def rowcol_to_latlon_rdr(row, col, lat_arr=None, lon_arr=None, geom_dir=None):
    """Look up the lat/lon for a row/col in radar coordinates (azimuth/range index)

    Args:
        row (int): row (azimuth index) of point of interest
        col (int): column (range index) of point of interest
        lat_arr (ndarray, optional): lat geometry array for radar coordinates.
        lon_arr (ndarray, optional): lon geometry array for radar coordinates.
        geom_dir (str, optional): directory containing the lat/lon arrays, if not passed

    Raises:
        ValueError: If none of lat_arr/lon_arr/geom_dir are provided

    Returns:
        (lat, col) for the (az, range) position of point closest to lat/lon
    """

    if lat_arr is None or lon_arr is None:
        import apertools.sario

        if geom_dir is None:
            raise ValueError("need either lat/lon geomaetry arrays or geom_dir")
        lat_arr, lon_arr = apertools.sario.load_rdr_latlon(geom_dir=geom_dir)
    return lat_arr[row, col], lon_arr[row, col]


def crop_rdr_by_bbox(
    bbox,
    rdr_image=None,
    rdr_file=None,
    lat_arr=None,
    lon_arr=None,
    geom_dir=None,
    **kwargs,
):
    """Crop an image in radar coordinates by a lat/lon bbox"""
    import numpy as np
    import apertools.sario

    if rdr_image is None:
        rdr_image = apertools.sario.load(rdr_file, use_gdal=True, **kwargs)

    if lat_arr is None or lon_arr is None:
        if geom_dir is None:
            raise ValueError("need either lat/lon geomaetry arrays or geom_dir")
        lat_arr, lon_arr = apertools.sario.load_rdr_latlon(geom_dir=geom_dir)

    left, bot, right, top = bbox
    mlon = np.logical_and(lon_arr < right, lon_arr > left)
    mlat = np.logical_and(lat_arr < top, lat_arr > bot)
    mask = mlon & mlat
    rows, cols = np.where(mask)
    rmin, rmax = np.min(rows), np.max(rows)
    cmin, cmax = np.min(cols), np.max(cols)

    img_crop = rdr_image.copy()
    img_crop[..., ~mask] = np.nan
    img_crop = img_crop[..., rmin:rmax, cmin:cmax]

    lat_crop = lat_arr.copy()
    lat_crop[~mask] = np.nan
    lat_crop = lat_crop[rmin:rmax, cmin:cmax]

    lon_crop = lon_arr.copy()
    lon_crop[~mask] = np.nan
    lon_crop = lon_crop[rmin:rmax, cmin:cmax]
    return img_crop, lat_crop, lon_crop
