#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
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
    outdir: str | Path=Path("."),
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
    parser.add_argument(
        "--orbit-dir", default="~/dev/orbits", help="Directory containing POEORB files"
    )
    args = parser.parse_args()

    iso_xml_files = list(Path(args.search_dir).glob("*.iso.xml"))
    logger.info(f"Found {len(iso_xml_files)} iso XML files.")

    # remove all suffixes, only product name left
    product_name_list = [p.name.split(".")[0] for p in iso_xml_files]

    # with ProcessPoolExecutor(max_workers=args.max_jobs) as exc:
    #     list(exc.map(download_safe_metadata, product_name_list))
    Parallel(n_jobs=args.max_jobs)(
        delayed(download_safe_metadata)(f, outdir=args.out_dir)
        for f in product_name_list
    )


if __name__ == "__main__":
    main()
