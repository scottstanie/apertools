#!/usr/bin/env python
"""
Makes scatterplots of the phase values vs elevation to see if linear trend exists
"""
import matplotlib.pyplot as plt

# import multiprocessing
# import os
import numpy as np

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


def plot_unw_vs_elevation():
    elpts = np.load("elevation_points.npy")[::1000]
    plt.figure()
    plt.scatter(elpts[:, 0], elpts[:, 1], s=0.1)
    plt.xlabel("elevation")
    plt.ylabel("unwrapped phase (rad)")
    plt.title("sample of all unwrapped igram points vs elevation")
    plt.show(block=True)


def subsample_dem(
    sub=5,
    stack_fname="unw_stack_20190101.h5",
    dem_fname="elevation_looked.dem",
    stack_dset="stack_flat_shifted",
):
    from apertools import sario
    import xarray as xr

    dem = sario.load(dem_fname)
    ds = xr.open_dataset(stack_fname)
    da = ds[stack_dset]
    ifg_stack = da.coarsen(lat=sub, lon=sub, boundary="trim").mean()
    dem_da = xr.DataArray(dem, coords={"lat": da.lat, "lon": da.lon})
    dem_da_sub = dem_da.coarsen(lat=sub, lon=sub, boundary="trim").mean()
    return dem_da_sub, ifg_stack


def linear_trend(x, y, mask_na):
    import xarray as xr

    # mask_na = np.logical_and(np.isnan(x), np.isnan(y))
    pf = np.polyfit(x[~mask_na], y[~mask_na], 1)
    # pf = np.polyfit(x[mask], y[mask], 1)
    return xr.DataArray(pf[0])


def plot_corr(dem_da_sub, ifg_stack, col_slices=[slice(None)], nbins=30, alpha=0.5):
    import xarray as xr
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, len(col_slices))
    for idx, col_slice in enumerate(col_slices):
        ax = axes[0, idx]
        dem_pixels = dem_da_sub[:, col_slice].stack(space=("lat", "lon"))
        ifg_pixels = ifg_stack[:, :, col_slice].stack(space=("lat", "lon"))
        print("ifg pixels shape:", ifg_pixels.shape)

        # # Option 1: get the correlation coefficients
        # trendvals = xr.corr(dem_pixels, ifg_pixels, dim="space")
        # bins = np.linspace(-1, 1, nbins)
        # trendvals.plot.hist(ax=ax, bins=bins, alpha=alpha)

        # Option 2: fit a line, find the slope of the line
        # mask_na = np.logical_and(np.isnan(x), np.isnan(y))
        # pf = np.polyfit(x[~mask_na], y[~mask_na], 1)
        ifg_mask = np.isnan(ifg_pixels)
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
        bins = nbins
        trendvals.plot.hist(ax=ax, bins=bins, alpha=alpha)

        # row 2: plot the phase vs elevation plot for one
        max_idx = np.abs(trendvals).argmax().item()
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
        axim = ax.imshow(ifg_stack[max_idx, :, col_slice])
        fig.colorbar(axim, ax=ax)


if __name__ == "__main__":
    # gdal_translate -outsize 2% 2% S1B_20190325.geo.vrt looked_S1B_20190325.geo.tif
    ifg_date_list = sario.find_igrams(".")
    unw_file_list = [
        f.replace(".int", ".unw") for f in sario.find_igrams(".", parse=False)
    ]
    # unw_file_list = [f.replace(".int", ".unwflat") for f in sario.find_igrams(".", parse=False)]
    save_unw_vs_elevation(unw_file_list)
    plot_unw_vs_elevation()
