import argparse
import os
import h5py
from apertools import latlon, sario
import numpy as np
import matplotlib.pyplot as plt


def pct_diff_larger(shift, mm_thresh, diff_patch):
    return np.nansum(np.abs(diff_patch + shift) > mm_thresh) / np.sum(
        ~np.isnan(diff_patch)
    )


def find_min_shift(diff_patch, mm_thresh=5):
    minshift, minval = 0, 999
    for shift in np.linspace(-5, 5):
        cur_val = pct_diff_larger(shift, mm_thresh, diff_patch)
        if cur_val < minval:
            minval = cur_val
            minshift = shift
    return minshift, float(minval)


def plot_stuff(
    asc_patch, desc_patch, axes=None, colorbar=True, cmap="seismic_wide", vm=20
):
    diff_patch = asc_patch - desc_patch

    if axes is None:
        plt.figure()
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
        ax3 = plt.subplot(133, sharex=ax1, sharey=ax1)
    else:
        ax1, ax2, ax3 = axes

    ax1.imshow(desc_patch, vmax=vm, vmin=-vm, cmap=cmap)
    ax2.imshow(asc_patch, vmax=vm, vmin=-vm, cmap=cmap)
    # axes[0].imshow(desc_patch, cmap='seismic')
    # axes[1].imshow(asc_patch, cmap='seismic')
    axim = ax3.imshow(diff_patch, vmax=vm, vmin=-vm, cmap=cmap)
    if colorbar:
        plt.colorbar(axim, ax=ax3)
    plt.show()
    return [ax1, ax2, ax3]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ascending")
    p.add_argument("descending")
    p.add_argument("--dset", default="velos/1")
    args = p.parse_args()
    dset = args.dset

    # desc_path = "/data4/scott/path85/allpath85/igrams/deformation_linear_maxtemp400_huber.h5"
    # asc_path = "deformation_linear_maxtemp400_huber.h5"
    asc_path = os.path.abspath(args.ascending)
    desc_path = os.path.abspath(args.descending)
    print("Ascending: %s, descending: %s" % (asc_path, desc_path))
    with h5py.File(asc_path) as f:
        asc_velo = f[dset][:]
        asc_mask = asc_velo == 0
        asc_velo[asc_mask] = np.nan
        asc_velo = latlon.LatlonImage(
            data=asc_velo, dem_rsc_file=sario.find_rsc_file(asc_path)
        )

    with h5py.File(desc_path) as f:
        desc_velo = f[dset][:]
        desc_mask = desc_velo == 0
        desc_velo[desc_mask] = np.nan
        desc_velo = latlon.LatlonImage(
            data=desc_velo, dem_rsc_file=sario.find_rsc_file(desc_path)
        )
        # desc_los = latlon.LatlonImage(data=desc_los_up, dem_rsc=desc_velo.dem_rsc)

    # TODO: still need to get better loading of defos vs velocities
    # desc_full, desc_mask = latlon.load_cropped_masked_deformation(full_path=desc_path, n=1)
    # desc_full = desc_full[0]

    # asc_full, asc_mask = latlon.load_cropped_masked_deformation(full_path=asc_path, n=1)
    # asc_full = asc_full[0]

    # asc_geolist = sario.load_geolist_from_h5(asc_path, dset=dset)
    # desc_geolist = sario.load_geolist_from_h5(desc_path, dset=dset)

    # asc_diff = (asc_geolist[-1] - asc_geolist[0]).days
    # desc_diff = (desc_geolist[-1] - desc_geolist[0]).days

    # asc_velo = asc_full / asc_diff * 365 * 10
    # desc_velo = desc_full / desc_diff * 365 * 10

    desc_los_file = os.path.join(os.path.split(desc_path)[0], "los_map.h5")
    asc_los_file = os.path.join(os.path.split(asc_path)[0], "los_map.h5")

    with h5py.File(asc_los_file) as f:
        asc_los_up = -f["stack"][:][2]  # ENU means up is 2
    with h5py.File(desc_los_file) as f:
        # with h5py.File("los_map.h5") as f:
        desc_los_up = -f["stack"][:][2]

    # asc_los = latlon.LatlonImage(data=asc_los_up, dem_rsc=asc_velo.dem_rsc)
    # desc_los = latlon.LatlonImage(data=desc_los_up, dem_rsc=desc_velo.dem_rsc)

    asc_velo_up = asc_velo / asc_los_up
    desc_velo_up = desc_velo / desc_los_up

    left, right, bottom, top = latlon.intersection_corners(
        asc_velo.dem_rsc, desc_velo.dem_rsc
    )
    print(left, right, bottom, top)

    # asc_patch = asc_velo[32.3:30.71, -104.1:-102.31]
    # desc_patch = desc_velo[32.3:30.71, -104.1:-102.31]
    asc_patch = asc_velo_up[top:bottom, left:right]
    desc_patch = desc_velo_up[top:bottom, left:right]
    diff_patch = asc_patch - desc_patch

    mm_thresh = 5
    shift, _ = find_min_shift(diff_patch, mm_thresh=mm_thresh)
    # Only look at left edge to minimize
    # shift, _ = find_min_shift(diff_patch[:, 50:500], mm_thresh=mm_thresh)
    # shift, _ = find_min_shift(diff_patch[:, 400:1000], mm_thresh=mm_thresh)
    # shift, _ = find_min_shift(diff_patch[:, 10:100], mm_thresh=mm_thresh)
    # shift = 1

    print(
        " %.2f is greater than %f mm/year difference on the two paths"
        % (pct_diff_larger(shift, mm_thresh, diff_patch), mm_thresh)
    )

    # asc_patch += shift
    # diff_patch += shift

    ax1, ax2, ax3 = plot_stuff(asc_patch + shift, desc_patch)
    # ax1, ax2, ax3 = plot_stuff(asc_patch, desc_patch - shift)
    plt.show()
