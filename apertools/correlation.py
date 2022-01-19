"""Module for exploting the full coherence/correlation matrix per pixel
"""
from datetime import datetime
import itertools
import numpy as np
import xarray as xr
import apertools.sario as sario
import apertools.utils as utils
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# TODO: lat, lon...
def create_cor_matrix(
    row_col=None,
    lat_lon=None,
    cor_filename="cor_stack.h5",
    ifg_filename="ifg_stack.h5",
    fill_val=0.0,
    window=1,
    dset="stack",
    min_date=datetime(2014, 1, 1),
    max_date=datetime(2022, 1, 1),
    slclist_ignore_file="slclist_ignore.txt",
    max_bandwidth=None,
    max_temporal_baseline=None,
    upper_triangle=False,
):
    with xr.open_dataset(cor_filename, engine="h5netcdf") as ds_cor, xr.open_dataset(
        ifg_filename, engine="h5netcdf"
    ) as ds_ifg:
        if row_col is not None:
            row, col = row_col
            cor_values = utils.window_stack_xr(
                ds_cor[dset], row=row, col=col, window_size=window
            )
            ifg_values = utils.window_stack_xr(
                ds_ifg[dset], row=row, col=col, window_size=window
            )
        elif lat_lon is not None:
            lat, lon = lat_lon
            cor_values = utils.window_stack_xr(
                ds_cor[dset], lon=lon, lat=lat, window_size=window
            )
            ifg_values = utils.window_stack_xr(
                ds_ifg[dset], lon=lon, lat=lat, window_size=window
            )
        else:
            raise ValueError("Must specify row_col or lat_lon")

    # ifglist = np.array(sario.parse_ifglist_strings(filenames))
    # slclist = np.array(utils.slclist_from_igrams(ifglist))
    # full_igram_list = np.array(utils.full_igram_list(slclist))

    slclist, ifglist = sario.load_slclist_ifglist(h5file=cor_filename)
    slclist, ifglist, valid_idxs = utils.filter_slclist_ifglist(
        ifg_date_list=ifglist,
        min_date=min_date,
        max_date=max_date,
        slclist_ignore_file=slclist_ignore_file,
        max_bandwidth=max_bandwidth,
        max_temporal_baseline=max_temporal_baseline,
    )
    cor_values = cor_values[valid_idxs]
    ifg_values = ifg_values[valid_idxs]

    full_igram_list = np.array(utils.full_igram_list(slclist))
    valid_idxs = np.searchsorted(
        sario.ifglist_to_filenames(full_igram_list),
        sario.ifglist_to_filenames(ifglist),
    )

    full_cor_values = fill_val * np.ones((len(full_igram_list),), dtype=np.complex64)
    full_cor_values[valid_idxs] = cor_values * np.exp(1j * np.angle(ifg_values))

    # Now arange into a matrix: fill the upper triangle
    ngeos = len(slclist)
    out = np.full((ngeos, ngeos), fill_val, dtype=np.complex64)

    # Diagonal is always 1
    rdiag, cdiag = np.diag_indices(ngeos)
    out[rdiag, cdiag] = 1

    rows, cols = np.triu_indices(ngeos, k=1)
    out[rows, cols] = full_cor_values
    if not upper_triangle:
        out = out + out.conj().T
        out[rdiag, cdiag] = 1
    return out, slclist, ifglist, cor_values


# TODO: plot with dates
def plot_cor_matrix(corrmatrix, slclist, vmax=None, vmin=0):
    coh = np.abs(corrmatrix)
    if vmax is None:
        # Make it slightly different color than the 1s on the diag
        vmax = 1.05 * np.nanmax(np.triu(coh, k=1))

    # Source for plot: https://stackoverflow.com/a/23142190
    x_lims = y_lims = mdates.date2num(slclist)

    fig, ax = plt.subplots()
    axim = ax.imshow(
        coh,
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

    Returns: Sigma, the (M, M) covariance matrix, where M is the # of SAR dates
        (unique dates in `ifg_date_list`)
    """
    M = len(ifg_date_list)
    sar_date_list = list(sorted(set(itertools.chain.from_iterable(ifg_date_list))))
    N = len(sar_date_list)

    if np.isscalar(sar_date_variances):
        sar_date_variances = sar_date_variances * np.ones(N)
    else:
        assert len(sar_date_list) == len(sar_date_variances)

    Sigma = np.zeros((M, M))
    for (colidx, ig2) in enumerate(ifg_date_list):
        for (rowidx, ig1) in enumerate(ifg_date_list):
            if colidx > rowidx:
                Sigma[rowidx, colidx] = Sigma[colidx, rowidx]
                continue  # symmetric, so just copy over

            d11, d12 = ig1
            d21, d22 = ig2
            # On the diagonal, the variances of the two sar dates are added
            if rowidx == colidx:
                assert ig1 == ig2
                sigma1 = sar_date_variances[sar_date_list.index(d11)]
                sigma2 = sar_date_variances[sar_date_list.index(d12)]
                Sigma[rowidx, colidx] = sigma1 + sigma2
                continue

            # If there's a matching date with same sign -> positive variance
            if d11 == d21:
                Sigma[rowidx, colidx] = sar_date_variances[sar_date_list.index(d11)]
            elif d12 == d22:
                Sigma[rowidx, colidx] = sar_date_variances[sar_date_list.index(d12)]
            # reverse the sign case
            elif d11 == d22:
                Sigma[rowidx, colidx] = -sar_date_variances[sar_date_list.index(d11)]
            elif d12 == d21:
                Sigma[rowidx, colidx] = -sar_date_variances[sar_date_list.index(d12)]
            # otherwise there's no match, leave as 0

    return Sigma


def get_cor_per_day(slclist, ifglist):
    from insar import ts_utils

    A = ts_utils.build_A_matrix(slclist, ifglist)

    first_day_rows = np.sum(A, axis=1) == 1
    AC = np.hstack((np.zeros((len(A), 1)), np.abs(A)))
    AC[first_day_rows, 0] = 1.0
    AC /= 2

    # CC, slclist_C, ifglist_C, cor_values = create_cor_matrix(
    #     lat_lon=(lat1, lon1),
    #     cor_filename=cropped_dir + "cor_stack.h5",
    #     ifg_filename=cropped_dir + "ifg_stack.h5",
    #     slclist_ignore_file=cropped_dir + "slclist_ignore.txt",
    # #     max_temporal_baseline=500,
    #     upper_triangle=True,
    # )
    # cor_per_day = np.linalg.lstsq(AC, cor_values)[0]


def get_cor_mask(
    cor_image,
    cor_thresh,
    smooth=True,
    winfrac=10,
    return_smoothed=False,
    strel_size=3,
):
    """Get a mask of the correlation values in `cor_image`

    Args:
        cor_image (np.ndarray): 2D array of correlation values
        cor_thresh (float): threshold for correlation values
        smooth (bool, optional): whether to apply a gaussian filter before
        masking. Removes any long-wavelength trend in correlation. Defaults to True.

    Returns:
        np.ndarray: 2D boolean array of same shape as `image`
    """
    import scipy.ndimage as ndi
    from skimage.morphology import disk, opening

    if np.isscalar(cor_thresh):
        cor_thresh = [cor_thresh]

    cor_image = cor_image.copy()
    cormask = np.logical_or(np.isnan(cor_image), cor_image == 0)
    if smooth:
        sigma = min(cor_image.shape) / winfrac
        cor_smooth = np.fft.ifft2(
            ndi.fourier_gaussian(
                # cor_smooth = ndi.filters.gaussian_filter(
                np.fft.fft2(cor_image),
                # cor_image,
                sigma=sigma,
            )
        ).real
        cor_image -= cor_smooth
    cor_image[cormask] = 0
    cor_image -= np.min(cor_image)
    cor_image[cormask] = 0
    masks = []
    for c in cor_thresh:
        mask = cor_image < c
        selem = disk(strel_size)
        # mask = closing(mask, selem) # Makes the mask bigger
        mask = opening(mask, selem)

        masks.append(mask)
    mask = np.stack(masks) if len(cor_thresh) > 1 else masks[0]

    return (mask, cor_smooth) if return_smoothed else mask


def fill_cvx(img, mask, max_iters=1500):
    """Fill in masked pixels in `img` using TV convex optimization"""
    import cvxpy as cp

    U = cp.Variable(shape=img.shape)
    obj = cp.Minimize(cp.tv(U))
    constraints = [cp.multiply(~mask, U) == cp.multiply(~mask, img)]
    prob = cp.Problem(obj, constraints)
    # Use SCS to solve the problem.
    prob.solve(verbose=True, solver=cp.SCS, max_iters=max_iters)
    print("optimal objective value: {}".format(obj.value))
    return prob, U
