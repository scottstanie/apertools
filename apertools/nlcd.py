"""Module for downloading and processing NLCD data."""

import numpy as np
from pathlib import Path
from pydantic import NonNegativeFloat
import requests
from xml.etree import ElementTree
import shutil
import subprocess
from . import utils, subset
from .log import logger

URL = "https://s3-us-west-2.amazonaws.com/mrlc/nlcd_2019_land_cover_l48_20210604.zip"


def download_nlcd(save_dir=None):
    out_folder = _get_nlcd_folder(save_dir)
    if out_folder.exists():
        logger.info("NLCD already downloaded")
        return out_folder
    logger.info(
        f"Downloading NLCD to {save_dir}. This may take a while... (>1 GB file)"
    )
    out_filepath = _download_big_file(URL, save_dir)
    logger.info(f"Unzipping NLCD to {out_folder}")
    _unzip_file(out_filepath, out_folder)
    # Clear out the .zip after unzipping
    logger.info("Removing .zip file")
    out_filepath.unlink()

    return out_folder


def load_nlcd(bbox, nlcd_folder=None, out_fname=""):
    """Load a subset of the NLCD data. Optionally save to `out_fname`."""
    if nlcd_folder is None:
        nlcd_folder = _get_nlcd_folder()
    img = list(Path(nlcd_folder).glob("*img"))[0]
    if Path(out_fname).exists():
        logger.info(f"{out_fname} already exists, loading")
        import rasterio as rio

        with rio.open(out_fname) as src:
            return src.read(1)
    return subset.copy_vrt(img, out_fname=out_fname, bbox=bbox)


def get_nlcd_metadata(nlcd_folder=None):
    """Parse metadata file for the NLCD data.

    Source:
    https://github.com/cheginit/pygeohydro/blob/418a6c7af1f8941836890348274c6a68cc7bb614/pygeohydro/pygeohydro.py
    """
    if nlcd_folder is None:
        nlcd_folder = _get_nlcd_folder()
    xml_file = list(Path(nlcd_folder).glob("*xml"))[0]
    with open(xml_file) as f:
        root = ElementTree.fromstring(f.read())

    base_path = "eainfo/detailed/attr/attrdomv/edom"
    edomv = root.findall(f"{base_path}/edomv")
    edomvd = root.findall(f"{base_path}/edomvd")

    cover_classes = {}
    descriptors = {}
    for t, v in zip(edomv, edomvd):
        class_num = t.text
        class_name, _, description = v.text.partition("-")

        cover_classes[class_num] = class_name.strip()
        descriptors[class_num] = description.strip()

    clist = [
        i.split() for i in root.find("eainfo/overview/eadetcit").text.split("\n")[2:]
    ]
    colors = {
        int(c): (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1)
        for c, r, g, b in clist
    }
    colors[0] = (*colors[0][:3], 0)

    nlcd_meta = {
        "classes": cover_classes,
        "categories": {
            "Background": ("127",),
            "Unclassified": ("0",),
            "Water": ("11", "12"),
            "Developed": ("21", "22", "23", "24"),
            "Barren": ("31",),
            "Forest": ("41", "42", "43", "45", "46"),
            "Shrubland": ("51", "52"),
            "Herbaceous": ("71", "72", "73", "74"),
            "Planted/Cultivated": ("81", "82"),
            "Wetlands": ("90", "95"),
        },
        "descriptors": descriptors,
        "roughness": {
            "11": 0.001,
            "12": 0.022,
            "21": 0.0404,
            "22": 0.0678,
            "23": 0.0678,
            "24": 0.0404,
            "31": 0.0113,
            "41": 0.36,
            "42": 0.32,
            "43": 0.4,
            "45": 0.4,
            "46": 0.24,
            "51": 0.24,
            "52": 0.4,
            "71": 0.368,
            "72": np.nan,
            "81": 0.325,
            "82": 0.16,
            "90": 0.086,
            "95": 0.1825,
        },
        "colors": colors,
        # Scott initial estimate of correlations:
        # correlations
    }

    return nlcd_meta


def get_overlapping_nlcd(filename):
    """Get the NLCD classes that overlap with the given raster."""
    import rasterio as rio
    from osgeo import gdal

    with rio.open(filename) as src:
        bounds = src.bounds
        rows, cols = src.shape[-2:]
        crs = src.crs

    ext = utils.get_file_ext(filename)
    nlcd_filename = str(filename).replace(ext, "_nlcd.tif")
    # Load the NLCD data cropped to the same bounds
    load_nlcd(bbox=bounds, out_fname=nlcd_filename)
    # Resample the NLCD to the same resolution as the input
    # TODO: Do i care about the multilooking effect on classes?
    out_ds = gdal.Warp(
        "",
        nlcd_filename,
        format="VRT",
        dstSRS=crs,
        width=cols,
        height=rows,
    )
    out = out_ds.ReadAsArray()
    out_ds = None
    return out


def get_mean_correlations(cor, nlcd_img, nlcd_meta, min_cor=0.0, max_cor=0.8):
    from scipy import ndimage
    import pandas as pd

    inp = cor.copy()
    mask = np.logical_or(inp == 0, np.isnan(inp))

    nlcd = nlcd_img.copy()
    nlcd[mask] = 0  # Background, will be ignored in ndimage.meann
    inp[mask] = np.nan
    index = np.array(list(nlcd_meta["classes"].keys())).astype(int)
    df = pd.DataFrame(
        data={
            "class_num": index,
            "class_name": nlcd_meta["classes"].values(),
            "mean_cor": ndimage.mean(cor, nlcd, index=index),
        }
    )
    df = df.set_index("class_num")
    # Set background/unclassified to nan
    df.loc[0] = np.nan
    df.loc[127] = np.nan
    df["mean_cor_rescaled"] = rescale(df["mean_cor"], min_cor, max_cor)
    return df


def make_cor_image(cor, nlcd_img, nlcd_meta, min_cor=0.01, max_cor=0.8):
    df = get_mean_correlations(
        cor, nlcd_img, nlcd_meta, min_cor=min_cor, max_cor=max_cor
    )
    # breakpoint()
    out = np.zeros_like(cor)
    for class_num in df.index:
        c = df.loc[class_num]["mean_cor_rescaled"]
        indexes = np.where(nlcd_img == class_num)
        out[indexes] = c

    mask = np.logical_or(cor == 0, np.isnan(cor))
    mask = np.logical_or(out == 0, mask)
    out[mask] = np.nan
    return out


def rescale(data, min_val=0.01, max_val=0.8):
    """Rescale the correlation values to be between min_val and max_val.
    Can be used to offset the estimate bias (which should have, e.g., water as 0)"""
    return (
        max_val * ((data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data)))
        + min_val
    )


def plot(nlcd_img, nlcd_meta, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    cmap, norm, legend = cover_legends(nlcd_meta)
    # axim = ax.imshow(nlcd_img, cmap=cmap, norm=norm)
    axim = ax.imshow(nlcd_img, cmap=cmap)
    # axim = ax.pcolormesh(nlcd_img, cmap=cmap)
    fig.colorbar(axim, ax=ax)
    return fig, ax


def cover_legends(nlcd_meta):
    """Colormap (cmap) and their respective values (norm) for land cover data legends."""
    import contextlib
    from matplotlib.colors import ListedColormap, BoundaryNorm

    bounds = list(nlcd_meta["colors"])
    with contextlib.suppress(ValueError):
        # Remove the background class
        bounds.remove(127)

    cmap = ListedColormap(list(nlcd_meta["colors"].values()))
    norm = BoundaryNorm(bounds, cmap.N)
    levels = bounds + [100]
    return cmap, norm, levels


def _get_nlcd_folder(save_dir=None):
    """Get the NLCD folder from the cache or download it."""
    if save_dir is None:
        save_dir = Path(utils.get_cache_dir())
    filepath = save_dir / URL.split("/")[-1]
    out_folder = Path(str(filepath).replace(".zip", ""))
    return out_folder


def _download_big_file(url, save_dir):
    # https://stackoverflow.com/a/39217788/4174466
    local_filename = url.split("/")[-1]
    out_filename = save_dir / local_filename
    with requests.get(url, stream=True) as r:
        with open(out_filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    return out_filename


def _unzip_file(filepath, save_dir):
    """Unzips in place the .hgt files downloaded"""
    # -o forces overwrite without prompt, -d specifices unzip directory
    unzip_cmd = f"unzip -o -d {save_dir} {filepath}"
    logger.info(unzip_cmd)
    subprocess.check_call(unzip_cmd, shell=True)
