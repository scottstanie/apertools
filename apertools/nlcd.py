"""Module for downloading and processing NLCD data."""
import numpy as np
from pathlib import Path
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
    }

    return nlcd_meta


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
