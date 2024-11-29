#!/usr/bin/env python3
import datetime
import sys
from pathlib import Path
from typing import Optional, Union

import cartopy.crs as ccrs
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
import rioxarray
import xarray as xr
from cartopy.io import img_tiles
from matplotlib.gridspec import GridSpec
from opera_utils import get_dates

from apertools import plotting
from apertools.plotting import padded_extent
from apertools.colors import DEFAULT_CMAP

Filename = Union[str, Path]


def open_stack(
    filenames: list[Filename],
    rechunk_by_time: bool = True,
) -> xr.DataArray:
    """Open a stack of files as a single xarray dataset.

    Parameters
    ----------
    filenames : list[Filename]
        List of filenames to open.
    rechunk_by_time : bool, default=True
        Whether to rechunk the dataset so that all times are in a single chunk.

    Returns
    -------
    xr.Dataset
        The dataset containing all files.
    """
    ds = (
        xr.open_mfdataset(
            filenames,
            preprocess=_prep,
            engine="rasterio",
            concat_dim="time",
            combine="nested",
        )
        .rename({"band_data": "displacement"})
        .drop_vars(["band"])
    )
    da = ds.displacement
    if rechunk_by_time:
        return da.chunk({"time": len(ds.time)})
    else:
        return da


def _prep(ds):
    """Preprocess individual dataset when loading with open_mfdataset."""
    fname = ds.encoding["source"]
    date = get_dates(fname)[1] if len(get_dates(fname)) > 1 else get_dates(fname)[0]
    if len(ds.band) == 1:
        ds = ds.sel(band=ds.band[0])
    return ds.expand_dims(time=[pd.to_datetime(date)])


# @click.command(context_settings=dict(help_option_names=["-h", "--help"]))
# @click.argument(
#     "timeseries_dir",
#     type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
# )
# @click.option(
#     "-v",
#     "--velocity-file",
#     type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
#     help="Path to velocity TIF file (default: timeseries_dir/velocity.tif)",
# )
# @click.option(
#     "-p",
#     "--poi",
#     callback=validate_point,
#     help="Point of interest as 'lat lon' or 'lat,lon'",
# )
# @click.option(
#     "--poi-rowcol",
#     is_flag=True,
#     help="Interpret POI coordinates as row/col instead of lat/lon",
# )
# @click.option(
#     "-r",
#     "--reference-point",
#     type=tuple[int, int],
#     help="Reference point as 'row col' or 'row,col'",
# )
# @click.option(
#     "--reference-file",
#     type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
#     help="File containing reference point coordinates (row,col)",
# )
# @click.option(
#     "-o",
#     "--output-file",
#     type=click.Path(dir_okay=False, path_type=Path),
#     help="Output file path (PNG/PDF/SVG)",
# )
# @click.option(
#     "--figsize",
#     type=tuple[float, float],
#     default=(12, 8),
#     callback=validate_point,
#     help="Figure size in inches as 'width height'",
# )
# @click.option(
#     "--zoom-level",
#     type=int,
#     default=9,
#     help="Zoom level for satellite imagery (default: 9)",
# )
# @click.option(
#     "--cmap",
#     type=str,
#     default="RdYlBu",
#     help="Colormap for velocity plot (default: RdYlBu)",
# )
# @click.option(
#     "--mask-zeros/--no-mask-zeros",
#     default=True,
#     help="Mask zero values in velocity data",
# )
# @click.option("--show/--no-show", default=True, help="Show the plot window")
def main(
    timeseries_dir: Path,
    velocity_file: Path,
    # poi: tuple[float, float],
    # poi_rowcol: bool,
    # reference_point: tuple[float, float],
    # reference_file: Path,
    output_file: Path,
    figsize: tuple[float, float],
    zoom_level: int = 8,
    cmap: str = DEFAULT_CMAP,
    mask_nodata: bool = True,
):
    """Create publication-ready InSAR time series plots using xarray.
    
    This tool generates a figure combining velocity maps and time series data
    from InSAR observations, overlaid on satellite imagery.
    
    Example usage:
    
    \b
    # Basic usage with lat/lon point of interest
    insar_plot timeseries/ -p "19.4 -156"
    
    \b
    # With row/col point of interest
    insar_plot timeseries/ --poi "100 200" --poi-rowcol
    
    \b
    # Full example
    insar_plot timeseries/ \\
        -v timeseries/velocity.tif \\
        -p "19.4 -156" \\
        -r "385 1180" \\
        -o figure.png \\
        --figsize "12 8" \\
        --zoom-level 10 \\
        --cmap RdYlBu
    """
    # Set default velocity file if not provided
    if velocity_file is None:
        velocity_file = timeseries_dir / "velocity.tif"
        if not velocity_file.exists():
            raise click.UsageError(
                f"No velocity file provided and {velocity_file} does not exist"
            )

    # Load velocity data
    click.echo("Loading velocity data...")
    velocity_ds = rioxarray.open_rasterio(velocity_file).rename({"band": "velocity"})
    velocity_latlon = velocity_ds.rio.reproject("EPSG:4326")

    # Load time series data
    click.echo("Loading time series data...")
    ts_pattern = str(timeseries_dir / "2*tif")
    ds = open_stack(ts_pattern)
    crs = ds.rio.crs
    _transformer_to_lonlat = Transformer.from_crs(crs, 4326, always_xy=True)
    _transformer_from_lonlat = Transformer.from_crs(4326, crs, always_xy=True)

    # Reproject to lat/lon
    # ts_latlon = ds.rio.reproject("EPSG:4326")

    # # Handle reference point from file if needed
    # if reference_file and not reference_point:
    #     try:
    #         with open(reference_file) as f:
    #             ref_text = f.read().strip()
    #             reference_point = tuple(map(int, ref_text.split(",")))
    #     except (ValueError, OSError) as e:
    #         raise click.UsageError(f"Error reading reference point file: {e}")

    fig, (ax_map, ax_ts) = plot_insar_timeseries(
        velocity_da=velocity_latlon,
        timeseries_da=ts_latlon.displacement,
        # poi=poi,
        # poi_is_latlon=not poi_rowcol,
        # reference_point=reference_point,
        # mask_zeros=mask_zeros,
        tile_zoom_level=zoom_level,
        figsize=figsize,
        cmap=cmap,
    )

    if output_file:
        click.echo(f"Saving figure to {output_file}...")
        fig.savefig(output_file, bbox_inches="tight", dpi=300)


def plot_insar_timeseries(
    velocity_da: xr.DataArray,
    timeseries_da: xr.DataArray,
    poi: Union[tuple[float, float], None] = None,
    poi_is_latlon: bool = True,
    reference_point: Union[tuple[int, int], None] = None,
    # mask: ArrayLike | None = True,
    tile_zoom_level: int = 9,
    figsize: tuple[float, float] = (12, 8),
    cmap: str = "RdYlBu",
    cbar_label: Optional[str] = None,
):
    """Create a publication-ready InSAR time series plot using xarray data.

    Parameters
    ----------
    velocity_da : xr.DataArray
        Velocity data in EPSG:4326 (lat/lon coordinates)
    timeseries_da : xr.DataArray
        Time series data in EPSG:4326
    poi : tuple, optional
        Point of interest as (lat, lon) if poi_is_latlon=True,
        or (row, col) if poi_is_latlon=False
    poi_is_latlon : bool, default=True
        Whether poi coordinates are in lat/lon or row/col
    reference_point : tuple, optional
        Reference point as (row, col)
    mask_zeros : bool, default=True
        Whether to mask zero values in velocity data
    tile_zoom_level : int, default=9
        Zoom level for background satellite imagery
    figsize : tuple, default=(12, 8)
        Figure size in inches
    cmap : str, default='RdYlBu'
        Colormap for velocity plot
    cbar_label : str, optional
        Label for colorbar. If None, tries to construct from metadata
    """
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[2, 1])

    # Top subplot: Map
    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())

    # Add satellite imagery
    tiler = img_tiles.GoogleTiles(style="satellite")
    ax_map.add_image(tiler, tile_zoom_level, interpolation="bicubic")

    # # Mask zeros if requested
    # if mask_zeros:
    #     velocity_plot = velocity_da.where(velocity_da != 0)
    # else:
    #     velocity_plot = velocity_da

    # # Get extent from velocity data
    # extent = padded_extent(velocity_da.rio.bounds(), 0.1)

    # # # Plot velocity data
    # im = ax_map.imshow(
    #     velocity_da,
    #     origin="upper",
    #     extent=extent,
    #     transform=ccrs.PlateCarree(),
    #     zorder=2,
    #     cmap=cmap,
    # )

    _, _, axim = plotting.map_background(image=velocity_da, ax=ax_map)

    # Bottom subplot: Time series
    ax_ts = fig.add_subplot(gs[1])
    # Plot time series
    dates = pd.to_datetime(timeseries_da.time.values)
    ts_data = np.zeros(len(dates))
    ts_line = ax_ts.plot(dates, ts_data, "o-", markersize=4)
    _transformer_to_lonlat = Transformer.from_crs(
        timeseries_da.rio.crs, 4326, always_xy=True
    )
    _transformer_from_lonlat = Transformer.from_crs(
        4326, timeseries_da.rio.crs, always_xy=True
    )

    def onclick(event):
        # Ignore right/middle click, clicks off image
        if not event.inaxes:
            return
        # button_press_event:
        # xy=(873, 464) xydata=(-155.4442863171439, 19.386708133972093)
        # button=3 dblclick=0 inaxes=< GeoAxes: +proj=eqc ... +type=crs >
        lon = event.xdata
        lat = event.ydata
        fig.suptitle(f"{lon}, {lat:.4f}")
        x, y = _transformer_from_lonlat.transform(lon, lat)
        # fig.suptitle(f"{x}, {y}")
        ts_data = timeseries_da.sel(x=x, y=y, method="nearest")
        ts_line.set_data(np.asarray(ts_data))
        # fig.suptitle(f"{np.array(ts_data)}")

        ts_line = ax_ts.plot(dates, ts_data, "o-", markersize=4)

    fig.canvas.mpl_connect("button_press_event", onclick)

    # if poi is not None:
    #     # Extract time series at point of interest
    #     if poi_is_latlon:
    #         lat, lon = poi
    #         ts_data = timeseries_da.sel(y=lat, x=lon, method="nearest")
    #     else:
    #         row, col = poi
    #         ts_data = timeseries_da.isel(y=row, x=col)

    #     # Plot time series
    #     ax_ts.plot(dates, ts_data, "o-", markersize=4)

    #     # TODO:
    #     # Plot point on map
    #     if poi_is_latlon:
    #         ax_map.plot(
    #             lon,
    #             lat,
    #             "r*",
    #             markersize=10,
    #             transform=ccrs.PlateCarree(),
    #             label="Point of Interest",
    #         )
    #     else:
    #         # Convert row/col to lat/lon
    #         lat = timeseries_da.y[row].item()
    #         lon = timeseries_da.x[col].item()
    #         ax_map.plot(
    #             lon,
    #             lat,
    #             "r*",
    #             markersize=10,
    #             transform=ccrs.PlateCarree(),
    #             label="Point of Interest",
    #         )

    # # Add reference point if provided
    # if reference_point:
    #     row, col = reference_point
    #     ref_lat = timeseries_da.y[row].item()
    #     ref_lon = timeseries_da.x[col].item()
    #     ax_map.plot(
    #         ref_lon,
    #         ref_lat,
    #         "k^",
    #         markersize=10,
    #         transform=ccrs.PlateCarree(),
    #         label="Reference",
    #     )
    #     ax_map.legend()

    # Format time series plot
    ax_ts.set_xlabel("Date")
    try:
        units = timeseries_da.attrs.get("units", "mm")
        ax_ts.set_ylabel(f"Displacement ({units})")
    except AttributeError:
        ax_ts.set_ylabel("Displacement")
    ax_ts.grid(True)

    # Rotate x-axis labels
    plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    return fig, (ax_map, ax_ts)


if __name__ == "__main__":
    main()
