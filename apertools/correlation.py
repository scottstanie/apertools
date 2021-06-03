"""Module for exploting the full coherence/correlation matrix per pixel
"""
import itertools
import numpy as np
import xarray as xr
import apertools.sario as sario
import apertools.utils as utils
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# TODO: lat, lon...
def create_cor_matrix(row_col=None, lat_lon=None, filename="cor_stack.nc", window=3):
    with xr.open_dataset(filename) as ds:
        filenames = ds.filenames
        win = window // 2
        if row_col is not None:
            row, col = row_col
        elif lat_lon is not None:
            lat, lon = lat_lon
            # corr_values = ds["stack"].sel(lat=lat, lon=lon, method="nearest")
            latarr, lonarr = ds.lat, ds.lon
            row = abs(latarr - lat).argmin()
            col = abs(lonarr - lon).argmin()
        cv_cube = ds["stack"][:, row - win : row + win + 1, col - win : col + win + 1]
        corr_values = cv_cube.mean(dim=("lat", "lon"))

    intlist = np.array(sario.parse_intlist_strings(filenames))
    geolist = np.array(utils.geolist_from_igrams(intlist))
    full_igram_list = np.array(utils.full_igram_list(geolist))

    valid_idxs = np.searchsorted(
        sario.intlist_to_filenames(full_igram_list),
        sario.intlist_to_filenames(intlist),
    )

    full_corr_values = np.nan * np.ones((len(full_igram_list),))
    full_corr_values[valid_idxs] = corr_values

    # Now arange into a matrix: fill the upper triangle
    ngeos = len(geolist)
    out = np.full((ngeos, ngeos), np.nan)

    # Diagonal is always 1
    rdiag, cdiag = np.diag_indices(ngeos)
    out[rdiag, cdiag] = 1

    rows, cols = np.triu_indices(ngeos, k=1)
    out[rows, cols] = full_corr_values
    return out, geolist


# TODO: plot with dates
def plot_corr_matrix(corrmatrix, geolist, vmax=None, vmin=0):
    if vmax is None:
        # Make it slightly different color than the 1s on the diag
        vmax = 1.05 * np.nanmax(np.triu(corrmatrix, k=1))

    # Source for plot: https://stackoverflow.com/a/23142190
    x_lims = y_lims = mdates.date2num(geolist)

    fig, ax = plt.subplots()
    axim = ax.imshow(
        corrmatrix,
        # Note: limits on y are reversed since top row is earliest
        extent=[x_lims[0], x_lims[-1], y_lims[-1], y_lims[0]],
        aspect="auto",
        vmax=vmax,
        vmin=vmin,
    )
    fig.colorbar(axim, ax=ax)
    ax.set_xlabel("Reference (early)")
    ax.set_ylabel("Secondary (late)")

    ax.xaxis_date()
    ax.yaxis_date()
    date_format = mdates.DateFormatter("%Y%m%d")

    ax.xaxis.set_major_formatter(date_format)
    ax.yaxis.set_major_formatter(date_format)
    # Tilt x diagonal:
    fig.autofmt_xdate()
    # fig.autofmt_ydate() # No y equivalent :(
    return fig, ax


def plot_bandwidth(ifg_dates):
    all_sar_dates = list(sorted(set(itertools.chain.from_iterable(ifg_dates))))
    nsar = len(all_sar_dates)
    # all_ifg_list = utils.full_igram_list(all_sar_dates)
    out = np.full((nsar, nsar), fill_value=True, dtype=bool)
    for idx in range(nsar):
        d1 = all_sar_dates[idx]
        for jdx in range(idx + 1, nsar):
            d2 = all_sar_dates[jdx]
            if (d1, d2) not in ifg_dates:
                out[idx, jdx] = out[jdx, idx] = False

    return out