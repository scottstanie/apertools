#!/usr/bin/env python
"""
Makes scatterplots of the phase values vs elevation to see if linear trend exists
"""
import matplotlib.pyplot as plt

# import multiprocessing
# import os
import numpy as np
import xarray as xr

from apertools import sario


def save_unw_vs_elevation(unw_file_list, every=50):
    points = np.empty((0, 2))
    elpts = sario.load("elevation_looked.dem")
    mask = ~np.isnan(sario.load(unw_file_list[0].replace(".unw", ".unwflat")))
    # elpts = elpts[:, :400]
    elpts = elpts[mask]
    elpts = elpts.reshape((-1,))

    for f in unw_file_list[::every]:
        out = sario.load(f)
        out -= np.mean(out)
        # new_pts = np.vstack((elpts, out[:, :400][mask].reshape((-1, )))).T
        new_pts = np.vstack((elpts, out[mask].reshape((-1,)))).T
        points = np.vstack((points, new_pts))
    np.save("elevation_points.npy", points)


def plot_phase_vs_elevation(elevation=None, phase=None, labels=None, ax=None):
    if elevation is None or phase is None:
        elpts = np.load("elevation_points.npy")[::1000]
        elevation, phase = elpts[:, 0], elpts[:, 1]
    if ax is None:
        fig, ax = plt.subplots()

    if labels is not None:
        for ii in np.unique(labels):
            idxs = labels == ii
            ax.scatter(elevation[idxs], phase[idxs], s=2, label=ii)
        ax.legend()
    else:
        ax.scatter(elevation.ravel(), phase.ravel(), s=0.1)
    ax.set_xlabel("elevation")
    ax.set_ylabel("unwrapped phase (rad)")
    ax.set_title("sample of all unwrapped igram points vs elevation")
    return ax


def subsample_dem(
    sub=5,
    stack_fname="unw_stack_20190101.h5",
    dem_fname="elevation_looked.dem",
    stack_dset="stack_flat_shifted",
    cor_fname="cor_stack_20190101.h5",
    cor_mean_dset="stack_mean",
):

    ds = xr.open_dataset(stack_fname)
    ifg_stack = ds[stack_dset]
    # Make DataArrays for the DEM and mean correlation image
    dem = xr.DataArray(
        sario.load(dem_fname), coords={"lat": ifg_stack.lat, "lon": ifg_stack.lon}
    )
    ds_cor = xr.open_dataset(cor_fname)
    cor = ds_cor[cor_mean_dset]
    if sub > 1:
        ifg_stack_sub = ifg_stack.coarsen(lat=sub, lon=sub, boundary="trim").mean()
        dem_sub = dem.coarsen(lat=sub, lon=sub, boundary="trim").mean()
        cor_sub = cor.coarsen(lat=sub, lon=sub, boundary="trim").mean()
        return dem_sub, ifg_stack_sub, cor_sub
    else:
        return dem, ifg_stack, cor


def fit_linear(x, y, mask=None):
    if mask is not None:
        xx, yy = x[~mask], y[~mask]
    else:
        xx, yy = x, y
    xx, yy = xx.ravel(), yy.ravel()
    mask_na = np.logical_or(np.isnan(xx), np.isnan(yy))
    xx, yy = xx[~mask_na], yy[~mask_na]

    pf = np.polyfit(xx, yy, 1)
    return xr.DataArray(pf)


def linear_trend(x, y, mask=None):
    pf = fit_linear(x, y, mask)
    return xr.DataArray(pf[0])


def plot_corr(
    dem_da_sub,
    ifg_stack_sub,
    col_slices=[slice(None)],
    nbins=30,
    alpha=0.5,
    cor_thresh=0.4,
    cor_mean_sub=None,
):

    # cormean = sario.load("cor_stack_20190101_mean.tif")
    # cor_mean_sub = xr.DataArray(
    #     cormean, coords={"lat": ifg_stack_sub.lat, "lon": ifg_stack_sub.lon}
    # )
    # ifg_stack_sub[0].data[(cor_mean_sub > cor_thresh)].shape

    fig, axes = plt.subplots(4, len(col_slices), squeeze=False)
    for idx, col_slice in enumerate(col_slices):
        ax = axes[0, idx]
        dem_pixels = dem_da_sub[:, col_slice].stack(space=("lat", "lon"))
        ifg_pixels = ifg_stack_sub[:, :, col_slice].stack(space=("lat", "lon"))
        print("ifg pixels shape:", ifg_pixels.shape)

        # Option 1: get the correlation coefficients
        trendvals = xr.corr(dem_pixels, ifg_pixels, dim="space")
        bins = np.linspace(-1, 1, nbins)

        # # Option 2: fit a line, find the slope of the line
        # x, y = dem_da_sub.data, ifg_stack_sub.data
        # mask_na = np.logical_and(np.isnan(x), np.isnan(y))
        # pf = np.polyfit(x[~mask_na], y[~mask_na], 1)

        # ifg_mask = (
        #     cor_mean_sub < cor_thresh
        #     if (cor_thresh and cor_mean_sub is not None)
        #     else None
        # )
        # trendvals = xr.apply_ufunc(
        #     linear_trend,
        #     dem_pixels,
        #     ifg_pixels,
        #     ifg_mask,
        #     vectorize=True,
        #     input_core_dims=[
        #         ["space"],
        #         ["space"],
        #         ["space"],
        #     ],  # reduce along "space", leaving 1 per ifg
        # )
        # trendvals *= 10  # Go from cm/meter of slope to mm/meter
        # bins = nbins

        trendvals.plot.hist(ax=ax, bins=bins, alpha=alpha)

        # row 2: plot the phase vs elevation plot for one
        max_idx = np.abs(trendvals).argmax().item()
        breakpoint()
        max_rho = trendvals[max_idx].item()
        ax = axes[1, idx]
        max_idx = np.abs(trendvals).argmax().item()
        ax.scatter(dem_pixels, ifg_pixels.isel(ifg_idx=max_idx).data.ravel())
        ax.set_title(r"$\rho = $" + "{:.2f}".format(max_rho))

        # row 3: plot the DEM
        ax = axes[2, idx]
        axim = ax.imshow(dem_da_sub[:, col_slice], cmap="gist_earth")
        fig.colorbar(axim, ax=ax)

        # row 3: plot the ifg with the strongest phase vs elevation trend
        ax = axes[3, idx]
        axim = ax.imshow(ifg_stack_sub[max_idx, :, col_slice])
        ax.set_title(f"ifg_idx = {max_idx}")
        fig.colorbar(axim, ax=ax)


def fit_elevation_phase(dem_da_sub, ifg_stack_sub, cor_mean_sub=None, cor_thresh=0.3):
    import xarray as xr

    dem_pixels = dem_da_sub.stack(space=("lat", "lon"))
    ifg_pixels = ifg_stack_sub.stack(space=("lat", "lon"))
    print("ifg pixels shape:", ifg_pixels.shape)

    ifg_mask = (
        cor_mean_sub < cor_thresh if (cor_thresh and cor_mean_sub is not None) else None
    )

    trendvals = xr.apply_ufunc(
        linear_trend,
        dem_pixels,
        ifg_pixels,
        ifg_mask,
        vectorize=True,
        input_core_dims=[
            ["space"],
            ["space"],
            ["space"],
        ],  # reduce along "space", leaving 1 per ifg
    )
    trendvals *= 10  # Go from cm/meter of slope to mm/meter
    return trendvals


def kmeans(
    dem_da_sub,
    ifg_stack_sub,
    ifg_idx=0,
    k=3,
    std_normalize=True,
    log_height=True,
    cor_thresh=0.3,
    cor_mean_sub=None,
):
    from sklearn.cluster import KMeans

    # from sklearn.cluster import SpectralClustering, DBSCAN
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    llon, llat = np.meshgrid(dem_da_sub.lon, dem_da_sub.lat)
    ifg = ifg_stack_sub[ifg_idx].data
    dem = dem_da_sub.data
    if cor_thresh and cor_mean_sub is not None:
        mask_cor = cor_mean_sub > cor_thresh
        ifg = ifg[mask_cor]
        dem = dem[mask_cor]
        llon = llon[mask_cor]
        llat = llat[mask_cor]
    ifg_pixels = ifg.ravel()
    dem_pixels = dem.ravel()
    goodidx = ~np.isnan(ifg_pixels)
    dem_pixels = dem_pixels[goodidx]
    ifg_pixels = ifg_pixels[goodidx]
    lon = llon.ravel()[goodidx]
    lat = llat.ravel()[goodidx]

    X = np.stack((dem_pixels, ifg_pixels, lon, lat)).T

    kmeans = KMeans(init="k-means++", n_clusters=k)
    # kmeans = SpectralClustering(n_clusters=k)
    # kmeans = DBSCAN(eps=.15, min_samples=50)
    if log_height:
        # Make the height log transformed
        Xn = X.copy()
        Xn[:, 0] = np.log(Xn[:, 0])

    if std_normalize:
        make_pipeline(StandardScaler(), kmeans).fit(X)
    else:
        kmeans.fit(X)

    try:
        centroids = kmeans.cluster_centers_
        print("Height centers of clusters:")
        print(centroids[:, 0])
        print("Lon, lat clusters:")
        print(centroids[:, 2:])
    except AttributeError:
        pass

    # Plot the phase/el, labelled by color
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    plot_phase_vs_elevation(X[:, 0], X[:, 1], labels=kmeans.labels_, ax=axes[0])

    plot_kmeans_latlon(kmeans, X, ax=axes[1])

    # last, plot the DEM and the ifg
    dem_da_sub.plot.imshow(ax=axes[2])
    ifg_stack_sub[ifg_idx].plot.imshow(ax=axes[3], cmap="RdBu")
    fig.tight_layout()
    return kmeans, fig, axes


def plot_kmeans_latlon(kmeans, X, ax=None):
    labels = kmeans.labels_
    if ax is None:
        fig, ax = plt.subplots()

    for ii in np.unique(labels):
        idxs = labels == ii
        ax.scatter(X[idxs, 2], X[idxs, 3], s=5, label=ii)
    ax.legend()

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Labels of clusters in space")
    return ax


def fit_halves(dem, phase, col=500, vm=2):
    poly_left = np.polyfit(dem.data[:, :col].ravel(), phase.data[:, :col].ravel(), 1)
    poly_right = np.polyfit(dem.data[:, col:].ravel(), phase.data[:, col:].ravel(), 1)
    plot_phase_vs_elevation(dem.data[:, :col].ravel(), phase.data[:, :col].ravel())
    plot_phase_vs_elevation(dem.data[:, col:].ravel(), phase.data[:, col:].ravel())
    print(poly_left, poly_right)
    ll, rr = np.polyval(poly_left, dem), np.polyval(poly_right, dem)
    ll[:, col:] = 0
    rr[:, :col] = 0
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(ll, cmap="seismic_wide_y_r", vmax=vm, vmin=-vm)
    axes[1].imshow(rr, cmap="seismic_wide_y_r", vmax=vm, vmin=-vm)
    return ll, rr

