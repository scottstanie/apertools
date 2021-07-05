import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import apertools.sario as sario
from apertools.deramp import remove_ramp
from apertools.constants import PHASE_TO_CM


def select_nearby_igrams(
    ifglist,
    day_lim=30,
    min_temporal=0,
    max_temporal=5000,
    month_range=range(1, 13),
):
    """Finds igrams withing `day_lim` days of each other, ignoring year
    Also can filter by temporal baseline
    month_range filters igrams with months in some range/set/list (e.g. [1, 2, 12] for DJF)
    """
    import pandas as pd

    int_df = pd.DataFrame(ifglist, columns=["day1", "day2"])
    int_df["month1"] = pd.DatetimeIndex(int_df["day1"]).month
    int_df["month2"] = pd.DatetimeIndex(int_df["day2"]).month
    int_df["ddiff"] = (int_df.day2 - int_df.day1) / pd.offsets.Day(1)
    # Get the year-agnostic day difference
    int_df["day_offset"] = ((int_df["ddiff"] + 180) % 365) - 180
    # int_df["wdiff"] = (int_df.day2 - int_df.day1) / pd.offsets.Week(1)
    # same_month_df = int_df[int_df.month1 == int_df.month2]
    close_dates = np.abs(int_df["day_offset"]) < day_lim
    too_near = int_df["ddiff"] < min_temporal
    too_long = int_df["ddiff"] > max_temporal
    # TODO: And? or Or?
    # month_in_range = int_df.month1.isin(month_range) & int_df.month2.isin(month_range)
    month_in_range = int_df.month1.isin(month_range) | int_df.month2.isin(month_range)

    good_igram_idxs = int_df[close_dates & ~too_near & ~too_long & month_in_range]

    return [ifglist[i] for i in good_igram_idxs.index]


def average_seasonal_igrams(
    ifg_dir=".",
    gi_file="slclist_ignore.txt",
    ext=".unw",
    day_lim=30,
    min_temporal=30,
    month_range=range(1, 13),
    normalize_by="total",  # means sum(phase_i) / sum(times_i)
    # normalize_by="per_date",  # means sum(phase_i / span_i ) / N
    to_cm=True,
    to_yearly_rate=True,
):
    slclist, ifglist = sario.load_slclist_ifglist(ifg_dir, slclist_ignore_file=gi_file)
    nearby_ifglist = select_nearby_igrams(
        ifglist,
        day_lim=day_lim,
        min_temporal=min_temporal,
        month_range=month_range,
    )
    print(f"Filtered {len(ifglist)} original igrams down to {len(nearby_ifglist)}")
    fnames = sario.ifglist_to_filenames(nearby_ifglist, ext=ext)

    out = np.zeros(sario.load(fnames[0]).shape)

    mask_fname = os.path.join(ifg_dir, "masks.h5")
    with h5py.File(mask_fname, "r") as f:
        mask_stack = f["igram"][:].astype(bool)

    out_mask = np.zeros_like(out).astype(bool)

    # Get masks for deramping
    mask_igram_date_list = sario.load_ifglist_from_h5(mask_fname)

    total_days = 0
    for (f, date_pair) in zip(fnames, nearby_ifglist):
        img = sario.load(f)
        baseline_days = (date_pair[1] - date_pair[0]).days
        if normalize_by == "per_date":
            img /= baseline_days
        elif normalize_by == "total":
            total_days += baseline_days

        out += img

        mask_idx = mask_igram_date_list.index(date_pair)
        out_mask |= mask_stack[mask_idx]

    if normalize_by == "per_date":
        out /= len(fnames)
    elif normalize_by == "total":
        out /= total_days

    out = remove_ramp(out, deramp_order=1, mask=out_mask)
    if to_cm:
        out *= PHASE_TO_CM
    if to_yearly_rate:
        out *= 365
    return out


def plot_results(stackavg):
    absmax = max(np.abs(np.nanmax(stackavg)), np.abs(np.nanmin(stackavg)))
    fig, ax = plt.subplots()
    axim = ax.imshow(
        stackavg, vmin=-0.8 * absmax, vmax=0.8 * absmax, cmap="seismic_wide_y"
    )
    fig.colorbar(axim)
    ax.set_title("LOS Yearly Velocity [cm/year]")
    return fig, ax


if __name__ == "__main__":
    fname = "slc_stack.nc"
    if os.path.exists(fname):
        data = xr.open_dataarray("slc_stack.nc")
    else:
        data = sario.save_slc_amp_stack(directory=".", ext=".slc", outname=fname)

    seasonal = data.groupby("date.season").mean()
    seasonal.plot(
        x="lon",
        y="lat",
        col="season",
        col_wrap=2,
        vmax=np.percentile(seasonal.data, 95),
        cmap="gray",
        #     cmap="discrete_seismic7",
    )