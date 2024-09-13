#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from pathlib import Path

import backoff
import pandas as pd
from asfsmd.core import download_annotations, make_patterns
from joblib import Parallel, delayed

from apertools.log import get_log

logger = get_log(name=__name__)

logging.getLogger("asfsmd").setLevel(logging.WARNING)


def download_safe_metadata(
    product_name: str,
    pol: str = "vv",
    outdir: str | Path = Path("."),
    skip_if_exists: bool = True,
):
    out_product = (Path(outdir) / product_name).with_suffix(".SAFE")
    if skip_if_exists and out_product.exists():
        logger.info(f"{product_name} exists. Skipping.")
        return

    logger.info(f"Downloading {product_name}")
    try:
        _download_safe_metadata(product_name, pol=pol, outdir=Path(outdir))
    except Exception:
        logger.error(f"Error downloading data from {product_name}", exc_info=True)


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def _download_safe_metadata(
    product_name: str,
    pol: str = "vv",
    outdir: Path = Path("."),
):
    """Use `asfsmd` to get the SAFE metadata for a product."""

    patterns = make_patterns(pol=pol)
    download_annotations([product_name], patterns=patterns, outdir=outdir)


def _get_product_list(search_dir: str):
    iso_xml_files = list(Path(search_dir).glob("*.iso.xml"))
    logger.info(f"Found {len(iso_xml_files)} iso XML files.")

    # remove all suffixes, only product name left
    return [p.name.split(".")[0] for p in iso_xml_files]


def _get_product_list_cmr(search_dir: str):
    safe_lists = Path(search_dir).glob("safes-*.txt")
    logger.info(f"Found {len(safe_lists)} text files of SAFE products.")
    return list(
        chain.from_iterable(path.read_text().splitlines() for path in safe_lists)
    )


def main() -> None:
    """Download Sentinel-1 metadata from a WKT file."""

    parser = argparse.ArgumentParser(
        description="Download S1 metadata from a WKT file."
    )
    parser.add_argument("search_dir", help="Location with .iso.xml files.")
    parser.add_argument("--out-dir", default=".", type=Path, help="Output directory.")
    parser.add_argument(
        "--max-jobs", type=int, default=10, help="Number of parallel calls to `asfsmd`"
    )
    args = parser.parse_args()

    product_name_list = _get_product_list(args.search_dir)
    if not product_name_list:
        product_name_list = _get_product_list_cmr(args.search_dir)
    if not product_name_list:
        raise ValueError(f"Found no products in {args.search_dir}")

    # with ProcessPoolExecutor(max_workers=args.max_jobs) as exc:
    #     list(exc.map(download_safe_metadata, product_name_list))
    Parallel(n_jobs=args.max_jobs)(
        delayed(download_safe_metadata)(f, outdir=args.out_dir)
        for f in product_name_list
    )


if __name__ == "__main__":
    main()
