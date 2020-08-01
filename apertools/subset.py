import numpy as np
import shapely.geometry
import rasterio as rio  # TODO: figure out if i want rio or gdal...
from apertools.log import get_log
from apertools.latlon import km_to_deg
logger = get_log()


def read_subset(bbox, in_fname, driver=None, bands=None):
    # bbox: left, bot, right, top
    with rio.open(in_fname, driver=driver) as src:
        w = src.window(*bbox)
        bands = bands or range(1, src.count + 1)
        # TODO: do i want to make this 2d if one band requested?
        return np.stack([src.read(b, window=w) for b in bands], axis=0)


def copy_subset(bbox, in_fname, out_fname, driver=None, bands=None, nodata=0, verbose=True):
    if verbose:
        logger.info(f"Subsetting {bbox} from {in_fname} to {out_fname}")
    img = read_subset(bbox, in_fname, driver)
    crs = get_crs(in_fname, driver=driver)
    transform = get_transform(in_fname, driver=driver, bbox=bbox)

    write_outfile(out_fname, img, crs=crs, transform=transform, driver=driver, nodata=nodata)


def write_outfile(out_fname,
                  img,
                  mode="w",
                  crs=None,
                  transform=None,
                  driver=None,
                  nodata=None,
                  dtype=None):
    if driver is None and out_fname.endswith(".tif"):
        driver = "GTiff"
    # TODO: do i wanna guess for any others?
    if driver is None:
        raise ValueError("'driver' is required to write dataset")

    if img.ndim < 3:
        img = img[np.newaxis, :, :]
    count = img.shape[0]
    dtype = dtype or img.dtype

    with rio.open(
            out_fname,
            mode,
            count=count,
            crs=crs,
            transform=transform,
            driver=driver,
            height=img.shape[1],
            width=img.shape[2],
            nodata=nodata,
            dtype=dtype,
    ) as dst:
        for band, layer in enumerate(img, start=1):
            dst.write(layer, band)


def get_intersection_bounds(fname1, fname2):
    with rio.open(fname1) as src1, rio.open(fname2) as src2:
        if src1.crs != src2.crs:
            raise ValueError(f"{fname1} has crs {src1.crs}, but {fname2} has csr {src2.crs}")
        b1 = shapely.geometry.box(*src1.bounds)
        b2 = shapely.geometry.box(*src2.bounds)
        return b1.intersection(b2).bounds


def get_transform(in_fname, driver=None, bbox=None):
    with rio.open(in_fname, driver=driver) as src:
        transform = src.window_transform(src.window(*bbox)) if bbox else src.transform
        return transform


def get_intersect_transform(fname1, fname2):
    int_bnds = get_intersection_bounds(fname1, fname2)
    return get_transform(fname1, bbox=int_bnds)


def get_crs(fname, driver=None):
    with rio.open(fname, driver=driver) as src:
        return src.crs


def get_driver(fname):
    with rio.open(fname) as src:
        return src.driver


def get_nodata(fname):
    with rio.open(fname) as src:
        return src.nodata


def read_intersections(fname1, fname2, band1=None, band2=None):
    bounds = get_intersection_bounds(fname1, fname2)
    with rio.open(fname1) as src1, rio.open(fname2) as src2:
        w1 = src1.window(*bounds)
        w2 = src2.window(*bounds)
        if band1 is None:
            r1 = np.stack([src1.read(n, window=w1) for n in range(1, src1.count + 1)], axis=0)
        else:
            r1 = src1.read(band1, window=w1)

        if band2 is None:
            r2 = np.stack([src2.read(n, window=w2) for n in range(1, src2.count + 1)], axis=0)
        else:
            r2 = src2.read(band2, window=w2)
        return r1, r2


def bbox_around_point(lon, lat, side_km=25):
    """ Finds (left, bot, right top) in deg around a lon, lat point"""
    side_deg = km_to_deg(side_km)
    r = side_deg / 2
    return (lon - r, lat - r, lon + r, lat + r)
