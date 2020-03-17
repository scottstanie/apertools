"""
Takes in coordinates, outputs bounds to use for dem download

Can come from a corner and height/width, or be used with
http://geojson.io to get a quick geojson polygon

Coordinates are (lon, lat) to match (x, y)
Bounding box order is  (left, bottom, right, top) (floats)
"""
from __future__ import division, print_function
import itertools
import shapely.geometry


def geojson_to_wkt(gj):
    return shapely.geometry.shape(gj).wkt


def bounding_box(geojson=None, top_corner=None, dlon=None, dlat=None):
    """From a geojson object, compute bounding lon/lats

    Note: either geojson required, OR top_corner, dlon, dlat required

    Valid geojson types: Geometry, Feature (Polygon), Feature Collection
        (will choose the first Feature in the collection)

    Args:
        geojson (dict): json pre-loaded into a dict
        top_corner (tuple[float, float]): top left corner of desired box
            as a (lon, lat)
        dlon (float): width of bounding box (if top_corner given)
        dlat (float): height of bounding box (if top_corner given)

    Returns:
        tuple[float]: the left,bottom,right,top coords of bounding box
    """

    if not geojson:
        if not top_corner or not dlon or not dlat:
            raise ValueError("Must provide geojson, or top_corner, dlon, and dlat")
        coordinates = corner_coords(top_corner=top_corner, dlon=dlon, dlat=dlat)
    else:
        coordinates = coords(geojson)

    left = min(float(lon) for (lon, lat) in coordinates)
    right = max(float(lon) for (lon, lat) in coordinates)

    top = max(float(lat) for (lon, lat) in coordinates)
    bottom = min(float(lat) for (lon, lat) in coordinates)
    return left, bottom, right, top


def corner_coords(top_corner=None, dlon=None, dlat=None, bot_corner=None):
    dlat = abs(dlat)
    if top_corner is not None:
        lon, lat = top_corner
    elif bot_corner is not None:
        # So that we can write the return function just one way
        lon, lat = bot_corner[0], bot_corner[1] + dlat
    else:
        raise ValueError("Need top_corner or bot_corner")

    return [
        [lon, lat],
        [lon + dlon, lat],
        [lon + dlon, lat - dlat],
        [lon, lat - dlat],
        [lon, lat],
    ]


def extent(geojson):
    """From a geojson object, compute bounding lon/lats

    Made to match the apertools.latlon.extent, which also matches
    `matplotlib.pyplot.imshow` keyword arg `extent`:

    Valid geojson types: Geometry, Feature (Polygon), Feature Collection
        (will choose the first Feature in the collection)

    Args:
        geojson (dict): json pre-loaded into a dict

    Returns:
        tuple[float]: the boundaries of the bounding box in order:
        (lon_left, lon_right, lat_bottom, lat_top)
    """
    left, bottom, right, top = bounding_box(geojson=geojson)
    return left, right, bottom, top


def corners_to_geojson(corners):
    """Takes in 5 points for the corners, returns geojson
    """
    return {"type": "Polygon", "coordinates": [corners]}


def coords(geojson):
    """Finds the coordinates of a geojson polygon

    Note: we are assuming one simple polygon with no holes

    Args:
        geojson (dict): loaded geojson dict

    Returns:
        list: coordinates of polygon in the geojson

    Raises:
        ValueError: if invalid geojson type (no 'geometry' in the json)
    """
    # First, if given a deeper object (e.g. from geojson.io), extract just polygon
    try:
        if geojson.get('type') == 'FeatureCollection':
            geojson = geojson['features'][0]['geometry']
        elif geojson.get('type') == 'Feature':
            geojson = geojson['geometry']
    except KeyError:
        raise ValueError("Invalid geojson")

    return geojson['coordinates'][0]


def format_coords(geojson_dict, decimals=4):
    """Prints out the lon,lat points in the polygon joined in one string

    Used for ASF API queries: https://www.asf.alaska.edu/get-data/learn-by-doing/
    E.g. (from their example api request, the following URL params are used)
    polygon=-155.08,65.82,-153.5,61.91,-149.50,63.07,-149.94,64.55,-153.28,64.47,-155.08,65.82

    Args:
        geojson (dict): json pre-loaded into a dict

    Returns:
        str: lon,lat points of the Polygon in order as 'lon1,lat1,lon2,lat2,...'
    """
    c = coords(geojson_dict)
    fmt_str = '{{0:.{decimals}f}}'.format(decimals=decimals)
    return ','.join(fmt_str.format(coord) for coord in itertools.chain.from_iterable(c))


def kml_string_fmt(gj_dict):
    # Example coord_string:
    # -102.2,29.5 -101.4,29.5 -101.4,28.8 -102.2,28.8 -102.2,29.5
    coord_list = coords(gj_dict)
    return ' '.join(map(lambda tup: ','.join((str(s) for s in tup)), coord_list))


if __name__ == '__main__':
    import sys
    import json
    try:
        gj_file = sys.argv[1]
    except IndexError:
        print("Usage: %s file.geojson" % sys.argv[0])

    with open(gj_file) as f:
        print(format_coords(json.load(f)))
