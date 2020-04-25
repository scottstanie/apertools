import rasterio as rio  # TODO: figure out if i want rio or gdal...
from apertools.log import get_log
logger = get_log()


def read_window(fname, bounds, band=1, driver=None):
    """Returns the window `bounds` (left, bot, right, top)"""
    with rio.open(fname, driver=driver) as src:
        return src.read(band, window=src.window(*bounds))


def read_subset(bbox, in_fname, driver=None):
    with rio.open(in_fname, driver=driver) as src:
        w = src.window(*bbox)
        return src.read(1, window=w)


def read_crs_transform(in_fname, driver=None):
    with rio.open(in_fname, driver=driver) as src:
        return src.crs, src.transform, src.count


def copy_subset(bbox, in_fname, out_fname, driver=None, bands=None, nodata=0, verbose=True):
    if verbose:
        logger.info(f"Subsetting {bbox} from {in_fname} to {out_fname}")
    img = read_subset(bbox, in_fname, driver)
    crs, transform, count = read_crs_transform(in_fname, driver)
    bands = bands or range(1, count + 1)

    with rio.open(
            out_fname,
            "w",
            crs=crs,
            transform=transform,
            driver=driver,
            height=img.shape[0],
            width=img.shape[1],
            count=len(bands),
            nodata=nodata,
            dtype=img.dtype,
    ) as dst:
        for b in bands:
            dst.write(img, b)