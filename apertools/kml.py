from __future__ import division, print_function
import subprocess
import os
import re
import rasterio as rio
from apertools import geojson, latlon

# TODO: GEOTiff stuff with gdal!
# Maybe rename this to something not kml, maybe use rasterio
# For box: upper left, lower right
# gdal_translate -of GTiff -a_ullr -105.19105 33.508629 -102.552689 31.8869 -a_srs EPSG:4326 20150726_20160720.tif out1.tif
#
# For quad_template
# gdal_translate -of GTiff -a_srs EPSG:4326 -gcp 1 1 -104.869621 33.508629 -gcp 1 2186 -105.191055 31.886913 -gcp 1482 1 -102.181068 33.105865 quick-look-1.png out_gcp.tif
#
# If we need to warp
# gdalwarp -s_srs EPSG:4326 -t_srs EPSG:3857 out1.tif out2.tif
#
# Reminder: 4326 is the WGS84 coordinate system, 3857 is pseudo mercator

# Square image in a box
box_template = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.2">
<GroundOverlay>
    <name> {title} </name>
    <description> {description} </description>
    <Icon>
          <href> {img_filename} </href>
    </Icon>
    <LatLonBox>
        <north> {north} </north>
        <south> {south} </south>
        <east> {east} </east>
        <west> {west} </west>
    </LatLonBox>
</GroundOverlay>
</kml>
"""

# One point as a push pin
point_template = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.2">
<Placemark>
    <name>{title}</name>
    <description>{description}</description>
    <styleUrl>#pushpin</styleUrl>
    <Point>
        <coordinates>{coord_string}</coordinates>
    </Point>
</Placemark>
</kml>
"""

# Generic polygon, from a geojson
polygon_template = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.2">
<Placemark id="mountainpin1">
    <name>{title}</name>
    <description>{description}</description>
    <styleUrl>#transRedPoly</styleUrl>
    <Polygon>
      <extrude>1</extrude>
      <altitudeMode>relativeToGround</altitudeMode>
      <outerBoundaryIs>
        <LinearRing>
        <coordinates>{coord_string}</coordinates>
        </LinearRing>
      </outerBoundaryIs>
    </Polygon>
</Placemark>
</kml>

"""
# This example from the Sentinel quick-look.png preview with map-overlay.kml
# Example coord_string:
# -102.2,29.5 -101.4,29.5 -101.4,28.8 -102.2,28.8 -102.2,29.5
quad_template = """\
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:gml="http://www.opengis.net/gml" xmlns:xfdu="urn:ccsds:schema:xfdu:1" xmlns:gx="http://www.google.com/kml/ext/2.2">
<GroundOverlay>
    <name>{title}</name>
    <description>{description}</description>
    <Icon>
        <href>{img_filename}</href>
    </Icon>
    <gx:LatLonQuad>
        <coordinates>{coord_string}</coordinates>
    </gx:LatLonQuad>
</GroundOverlay>
</kml>
"""


# TODO: this may be wrong? might be 1 pixel width shy
def rsc_nsew(rsc_data):
    """return tuple of (north, south, east, west) from rsc data"""

    north = rsc_data['y_first']
    west = rsc_data['x_first']
    east = west + rsc_data['width'] * rsc_data['x_step']
    south = north + rsc_data['file_length'] * rsc_data['y_step']
    return north, south, east, west


def rsc_bounds(rsc_data):
    """Uses the x/y and step data from a .rsc file to generate LatLonBox for .kml"""
    north, south, east, west = rsc_nsew(rsc_data)
    return {'north': north, 'south': south, 'east': east, 'west': west}


# def parse_quad_kml(quad_kml_filename):
#     lon_lat_overlay_coords = latlon.map_overlay_coords(self.map_overlay_kml)
#     etree = ElementTree.parse(quad_kml_filename)
#     root = etree.getroot()


def create_kml(rsc_data=None,
               img_filename=None,
               gj_dict=None,
               title=None,
               desc="Description",
               shape='box',
               kml_out=None,
               lon_lat=None):
    """Make a kml file to display a image (tif/png) in Google Earth

    Args:
        rsc_data (dict): dem rsc data
        img_filename (str): name of the image file
        title (str): Title for kml metadata
        desc (str): Description kml metadata
        shape (str): Options = ('box', 'quad'). Box is square, quad is arbitrary 4 sides
        kml_out (str): filename of kml to write
        lon_lat (tuple[float]): if shape == 'point', the lon and lat of the point
    """
    if title is None:
        title = img_filename

    valid_shapes = ('box', 'quad', 'point', 'polygon')
    if shape not in valid_shapes:
        raise ValueError("shape must be %s" % ', '.join(valid_shapes))

    if shape == 'box':
        output = box_template.format(title=title,
                                     description=desc,
                                     img_filename=img_filename,
                                     **rsc_bounds(rsc_data))
    elif shape == 'quad':
        output = quad_template.format(title=title,
                                      description=desc,
                                      img_filename=img_filename,
                                      coord_string=geojson.kml_string_fmt(gj_dict))
    elif shape == 'point':
        if lon_lat is None:
            # TODO: do we want to accept geojson? or overkill?
            raise ValueError("point must include lon_lat tuple")
        output = point_template.format(
            title=title,
            description=desc,
            coord_string='{},{}'.format(*lon_lat),
        )
    elif shape == 'polygon':
        if gj_dict is None:
            raise ValueError("polygon must include gj_dict tuple")
        output = polygon_template.format(
            title=title,
            description=desc,
            coord_string=geojson.kml_string_fmt(gj_dict),
        )

    if kml_out:
        print("Saving kml to %s" % kml_out)
        with open(kml_out, 'w') as f:
            f.write(output)

    return output


def create_geotiff(
    rsc_data=None,
    kml_file=None,
    img_filename=None,
    shape='box',
    outfile='out.tif',
):
    """Create geotiff from rsc_data and image file

    Args:
        rsc_data (dict): dem rsc data
        kml_file (str): name of Sentinel provided kml with coordinates for quick-look.png
        img_filename (str): name of the image file
        shape (str): Options = ('box', 'quad'). Box is square, quad is arbitrary 4 sides
        kml_out (str): filename of kml to write
        lon_lat (tuple[float]): if shape == 'point', the lon and lat of the point
    """

    gdal_translate_box = "gdal_translate -of GTiff -a_nodata 0 -a_srs EPSG:4326 -a_ullr {ullr} {input} {out}"
    gdal_translate_quad = "gdal_translate -of GTiff -a_nodata 0 -a_srs EPSG:4326 -gcp 1 1 {ul} -gcp 1 {nrows} {ll} -gcp {ncols} 1 {ur} {input} {out}"
    # gdalwarp -s_srs EPSG:4326 -t_srs EPSG:3857 out1.tif out2.tif
    if shape == 'box':
        # ullr means upper left, lower right (lon, lat) points
        north, south, east, west = rsc_nsew(rsc_data)
        ullr_string = "%f %f %f %f" % (west, north, east, south)
        cmd = gdal_translate_box.format(ullr=ullr_string, input=img_filename, out=outfile)
    elif shape == 'quad':
        # coords:
        # [(-105.191055, 31.886913),
        #  (-104.869621, 33.508629),
        #  (-102.552689, 31.481976),
        #  (-102.181068, 33.105865)]

        with rio.open(img_filename) as ds:
            ncols, nrows = ds.width, ds.height

        # TODO: prob gettable through gdal/rasterio too
        ll, ul, lr, ur = sorted(latlon.map_overlay_coords(kml_file),
                                key=lambda tup: (tup[0], -tup[1]))
        ul = '%s %s' % ul
        ll = '%s %s' % ll
        ur = '%s %s' % ur
        cmd = gdal_translate_quad.format(
            ul=ul,
            ll=ll,
            ur=ur,
            nrows=nrows,
            ncols=ncols,
            input=img_filename,
            out=outfile,
        )

    print('Running:')
    print(cmd)
    subprocess.check_call(cmd, shell=True)


def extract_raster_outline(filename, outfile=None, band=1):
    """Takes a raster and finds the outline, ignoring nodata (assumed 0), saves as .shp

    https://gis.stackexchange.com/q/120994
    """
    if outfile is None:
        outfile = filename + ".shp"
    print("Writing to %s" % outfile)

    # First, make into a binary image
    outtmp = "tmp.binary.tif"
    calc_cmd = """gdal_calc.py --outfile={outtmp} -A {fin} --NoDataValue=0 --calc="A>0" """.format(
        outtmp=outtmp, fin=filename)
    print("Finding nodata outline:")
    print(calc_cmd)
    subprocess.check_call(calc_cmd, shell=True)

    # Then extract polygon: -8 means 8-connected, -b 1 is use band #1
    poly_cmd = """gdal_polygonize.py -8 {tmp} -b 1 {out} """.format(tmp=outtmp, out=outfile)
    print("Extracting polygon with command:")
    print(poly_cmd)
    subprocess.check_call(poly_cmd.split())
    os.remove(outtmp)
