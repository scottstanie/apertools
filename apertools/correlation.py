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

    ifglist = np.array(sario.parse_ifglist_strings(filenames))
    slclist = np.array(utils.slclist_from_igrams(ifglist))
    full_igram_list = np.array(utils.full_igram_list(slclist))

    valid_idxs = np.searchsorted(
        sario.ifglist_to_filenames(full_igram_list),
        sario.ifglist_to_filenames(ifglist),
    )

    full_corr_values = np.nan * np.ones((len(full_igram_list),))
    full_corr_values[valid_idxs] = corr_values

    # Now arange into a matrix: fill the upper triangle
    ngeos = len(slclist)
    out = np.full((ngeos, ngeos), np.nan)

    # Diagonal is always 1
    rdiag, cdiag = np.diag_indices(ngeos)
    out[rdiag, cdiag] = 1

    rows, cols = np.triu_indices(ngeos, k=1)
    out[rows, cols] = full_corr_values
    return out, slclist


# TODO: plot with dates
def plot_corr_matrix(corrmatrix, slclist, vmax=None, vmin=0):
    if vmax is None:
        # Make it slightly different color than the 1s on the diag
        vmax = 1.05 * np.nanmax(np.triu(corrmatrix, k=1))

    # Source for plot: https://stackoverflow.com/a/23142190
    x_lims = y_lims = mdates.date2num(slclist)

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


# fig, ax = plt.subplots()
# ax.imshow(plot_bandwidth(ifg_list), vmin=0, vmax=1)
# ticks = np.arange(.5, len(slc_list) + .5)
# ax.set_xticks(ticks)
# ax.set_xticklabels([])
# ax.set_yticks(ticks)
# ax.set_yticklabels([])
# ax.grid(which='major', markevery=.5, color='k')
# In [93]: out = apertools.correlation.plot_bandwidth(utils.filter_slclist_ifglist(ifglist_full, max_temporal_baseline=300)[1])
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


def cov_matrix_tropo(ifg_date_list, sar_date_variances):
    """Create a covariance matrix for tropospheric noise for 1 pixel

    Args:
        ifg_date_list (Iterable[Tuple]): list of (early, late) dates for ifgs
        sar_date_variances (Iterable[Tuple]): list of variances per SAR date
            if a scalar is given, will use the same variance for all dates
    
    Returns: sigma, the (M, M) covariance matrix, where M is the # of SAR dates
        (unique dates in `ifg_date_list`)
    """
    M = len(ifg_date_list)
    sar_date_list = list(sorted(set(itertools.chain.from_iterable(ifg_date_list))))
    N = len(sar_date_list)

    if np.isscalar(sar_date_variances):
        sar_date_variances = sar_date_variances * np.ones(N)
    else:
        assert len(sar_date_list) == len(sar_date_variances)

    sigma = np.zeros((M, M))
    for (jdx, ig2) in enumerate(ifg_date_list):
        for (idx, ig1) in enumerate(ifg_date_list):
            if jdx > idx:
                sigma[idx, jdx] = sigma[jdx, idx]
                continue  # symmetric, so just copy over

            d11, d12 = ig1
            d21, d22 = ig2
            if idx == jdx:
                assert ig1 == ig2
                sigma1 = sar_date_variances[sar_date_list.index(d11)]
                sigma2 = sar_date_variances[sar_date_list.index(d12)]
                sigma[idx, jdx] = sigma1 + sigma2
                continue

            # If there's a matching date with same sign -> positive variance
            if d11 == d21:
                sigma1 = sar_date_variances[sar_date_list.index(d11)]
                sigma[idx, jdx] = sigma1
            elif d12 == d22:
                sigma2 = sar_date_variances[sar_date_list.index(d12)]
                sigma[idx, jdx] = sigma2
            # reverse the sign case
            elif d12 == d21 or d11 == d22:
                sigma2 = sar_date_variances[sar_date_list.index(d12)]
                sigma[idx, jdx] = -sigma2
            # otherwise there's no match, leave as 0

    return sigma
