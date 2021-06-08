from osgeo import gdal


def georeference_image(filename, rsc_filename="dem.rsc"):
    # from gdal_edit.py
    # https://github.com/OSGeo/gdal/blob/master/gdal/swig/python/scripts/gdal_edit.py
    ds = gdal.Open(filename)

    srs = gdal.osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    ds.SetProjection(srs.ExportToWkt())

    rsc_data = read_rsc(rsc_filename)
    ds.SetGeoTransform(rsc_to_geotransform(rsc_data))
    ds = None
    return


def read_rsc(rsc_filename="dem.rsc"):
    str_dict = dict(l.split() for l in open(rsc_filename).readlines())
    return {k.lower(): to_number(v) for k, v in str_dict.items()}


def to_number(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            pass
    return s


def rsc_to_geotransform(rsc_data):

    # See here for geotransform info
    # https://gdal.org/user/raster_data_model.html#affine-geotransform
    # NOTE: gdal standard is to reference pixel by top left corner,
    # while the SAR .rsc stuff wants center of pixel
    # Xgeo = GT(0) + Xpixel*GT(1) + Yline*GT(2)
    # Ygeo = GT(3) + Xpixel*GT(4) + Yline*GT(5)

    # So for us, this means we have
    # X0 = trans[0] + .5*trans[1] + (.5*trans[2])
    # Y0 = trans[3] + (.5*trans[4]) + .5*trans[5]
    # where trans[2], trans[4] are 0s for north-up rasters

    x_step = rsc_data["x_step"]
    y_step = rsc_data["y_step"]
    X0 = rsc_data["x_first"] - 0.5 * x_step
    Y0 = rsc_data["y_first"] - 0.5 * y_step
    return (X0, x_step, 0.0, Y0, 0.0, y_step)
