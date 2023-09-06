#!/usr/bin/env python
from __future__ import annotations

import argparse

# from concurrent.futures import ProcessPoolExecutor
import warnings
from pathlib import Path

import asf_search as asf
import backoff
import geopandas as gpd
import pandas as pd
import s1reader
from joblib import Parallel, delayed
from shapely.ops import unary_union
from tqdm.auto import tqdm


def get_north_america_geoms(
    shapefile="~/repos/burst_db/src/burst_db/data/north_america_opera.geojson.zip",
    simplify_deg=0.3,
):
    df = gpd.read_file(shapefile)
    all_geoms = []
    usa_geom = None
    for _, row in df.iterrows():
        cur_geom = row.geometry.buffer(simplify_deg / 2).simplify(simplify_deg)
        if row.ISO_A3 == "USA":  # US shape needs more work
            usa_geom = cur_geom
            continue
        all_geoms.append(cur_geom)

    usa_geom_list = sorted(usa_geom.geoms, key=lambda g: g.area, reverse=True)
    conus, alaska, *others = usa_geom_list
    centroids = [g.centroid for g in others]
    west_poly = unary_union([others[i] for i, c in enumerate(centroids) if c.x < 0])
    east_poly = unary_union([others[i] for i, c in enumerate(centroids) if c.x > 0])
    all_geoms.extend([west_poly, east_poly, alaska, conus])
    # from manually running asf_search, these came up with 0 results
    empty_geom_idxs = [3, 9, 11]
    # In [183]: asf.geo_search(
    #     ...:         intersectsWith= df.iloc[[3, 9, 11]].geometry.unary_union.wkt,
    #     ...:         platform=[asf.PLATFORM.SENTINEL1],
    #     ...:         maxResults=2,
    #     ...:         processingLevel=asf.SLC,
    #     ...:         beamMode=asf.BEAMMODE.IW,
    #     ...:     )
    # Out[183]: ASFSearchResults([])
    all_geoms = [g for i, g in enumerate(all_geoms) if i not in empty_geom_idxs]
    return all_geoms


def get_burst_border_approx(burst):
    return unary_union(burst.border).simplify(1 / 3600).wkt


def to_row(burst, safe_file):
    burst_id = burst.burst_id
    dt = burst.sensing_start.isoformat()
    border = get_burst_border_approx(burst)
    return burst_id, dt, border, Path(safe_file).stem


def get_burst_rows(
    safe_file: str | Path, out_dir: str | Path, orbit_dir="/home/staniewi/dev/orbits/"
):
    try:
        outfile = (Path(out_dir) / Path(safe_file).stem).with_suffix(".csv")
        if outfile.exists():
            return
        orbit_file = s1reader.get_orbit_file_from_dir(safe_file, orbit_dir=orbit_dir)
        all_rows = []
        for iw in [1, 2, 3]:
            bursts = s1reader.load_bursts(
                safe_file, orbit_file, swath_num=iw, flag_apply_eap=False
            )
            for b in bursts:
                all_rows.append(to_row(b, safe_file))

        pd.DataFrame(all_rows).to_csv(outfile, header=False, mode="w", index=False)
    except Exception as e:
        print(f"Failure on {safe_file}: {e}")
        outfile = (Path(out_dir) / f"failure_{Path(safe_file).stem}").touch()


def make_all_safe_metadata(
    *,
    out_dir: str | Path,
    dir_with_safes: str | Path | None,
    safe_list: list[str | Path] | None = None,
    orbit_dir="/home/staniewi/dev/orbits/",
    max_jobs=20,
):
    if safe_list is None:
        safe_list = sorted(Path(dir_with_safes).glob("*.SAFE"))
        print(f"Found {len(safe_list)} SAFE dirs in {dir_with_safes}")

    print(f"Writing CSVs to {out_dir}")

    warnings.filterwarnings("ignore", category=UserWarning)  # s1reader is chatty
    Parallel(n_jobs=max_jobs)(
        delayed(get_burst_rows)(f, out_dir=out_dir, orbit_dir=orbit_dir)
        for f in safe_list
    )


def download_geojson_metadata(
    out_dir: str,
    start_date: str,
    end_date: str,
    outfile: str = "results.gpkg",
    **kwargs,
):
    all_geoms = get_north_america_geoms()
    out = Path(out_dir) / outfile
    for g in tqdm(all_geoms, desc="Processing geometries"):
        if g.area < 5:
            nweeks = 2 * 52
        elif g.area < 10:
            nweeks = 52
        elif g.area < 20:
            nweeks = 26
        elif g.area < 200:
            nweeks = 4
        else:
            nweeks = 2
        date_range = pd.date_range(
            start=start_date, end=end_date, freq=f"{nweeks}W"
        ).strftime("%Y-%m-%dT%H:%M:%S")
        tqdm.write(f"Processing geometry with area {g.area}")

        for start, end in tqdm(
            list(zip(date_range[:-1], date_range[1:])), desc="Date range"
        ):
            tqdm.write(f"Downloading {start} to {end}")
            results = asf.geo_search(
                intersectsWith=g.wkt,
                platform=[asf.PLATFORM.SENTINEL1],
                maxResults=2000,
                processingLevel=asf.SLC,
                beamMode=asf.BEAMMODE.IW,
                polarization=asf.POLARIZATION.VV,
                start=start,
                end=end,
            )
            if len(results) == 0:
                continue
            add_to_file(results, out)


def add_to_file(results: asf.ASFProduct, out: Path):
    rdf = gpd.GeoDataFrame.from_features(results.geojson())
    mode = "a" if out.exists() else "w"
    rdf.to_file(out, mode=mode)


def download_iso_metadata(
    wkt_file: str,
    out_dir: str,
    start_date: str,
    end_date: str,
    **kwargs,
):
    biweekly_dates = pd.date_range(start=start_date, end=end_date, freq="2W").strftime(
        "%Y-%m-%d"
    )

    for start, end in tqdm(list(zip(biweekly_dates[:-1], biweekly_dates[1:]))):
        tqdm.write(f"Downloading {start} to {end}")
        try:
            download_with_retry(
                out_dir=out_dir,
                wkt_file=wkt_file,
                start=start,
                end=end,
                processingLevel="METADATA_SLC",
                maxResults=2000,
                platform="S1",
                beamMode="IW",
            )
        except Exception as e:
            tqdm.write(f"Error downloading data from {start} to {end}: {e}")


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def download_with_retry(
    out_dir: str,
    wkt_file: str,
    start: str,
    end: str,
    processingLevel: str,
    maxResults: int,
    platform: str,
    beamMode: str,
) -> None:
    from apertools import asfdownload

    asfdownload.download_data(
        out_dir=out_dir,
        wkt_file=wkt_file,
        start=start,
        end=end,
        processingLevel=processingLevel,
        maxResults=maxResults,
        platform=platform,
        beamMode=beamMode,
    )


def main() -> None:
    """Download Sentinel-1 metadata from a WKT file."""

    parser = argparse.ArgumentParser(
        description="Download S1 metadata over a region of interest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wkt-file", help="Input WKT file (for ISO search).")
    parser.add_argument(
        "--iso",
        action="store_true",
        help="Download the .iso.xml metadata. Otherwise, gets the ASF search results as geojson",
    )
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument(
        "--outfile",
        default="results.gpkg",
        help="For GeoJson results, name of GPKG file to store results into.",
    )
    parser.add_argument(
        "--start-date", default="2014-09-01", help="Start date in YYYY-mm-dd format."
    )
    parser.add_argument(
        "--end-date",
        default=str(pd.Timestamp.today().date()),
        help="End date in YYYY-mm-dd format.",
    )

    args = parser.parse_args()
    arg_dict = vars(args)

    if args.iso:
        if args.wkt_file is None:
            raise ValueError("--wkt-file required for --iso")
        download_iso_metadata(**arg_dict)
    else:
        download_geojson_metadata(**arg_dict)


if __name__ == "__main__":
    main()
