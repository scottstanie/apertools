# coding: utf-8
from datetime import date
import numpy as np
import apertools.sario as sario
import apertools.subset as subset
from insar.prepare import remove_ramp
from insar.timeseries import PHASE_TO_CM

MENTONE_EQ_DATE = date(2020, 3, 26)


def stack_igrams(
    event_date=MENTONE_EQ_DATE,
    num_igrams=None,
    use_cm=True,
    rate=False,
    outname=None,
    verbose=True,
    ref=(5, 5),
    window=5,
    ignore_geos=True,
    cc_thresh=None,
    avg_cc_thresh=0.35,
    sigma_filter=0.3,
):

    gi_file = "geolist_ignore.txt" if ignore_geos else None
    geolist, intlist = sario.load_geolist_intlist(".", geolist_ignore_file=gi_file)
    # stack_igrams = select_igrams(geolist, intlist, event_date, num_igrams=num_igrams)
    stack_igrams = select_pre_event(geolist, intlist, event_date)
    # stack_igrams = select_post_event(geolist, intlist, event_date)

    stack_fnames = sario.intlist_to_filenames(stack_igrams, ".unw")
    if verbose:
        print("Using the following igrams in stack:")
        for f in stack_fnames:
            print(f)

    dts = [(pair[1] - pair[0]).days for pair in stack_igrams]

    cur_phase_sum, cc_stack = create_stack(
        stack_igrams,
        stack_fnames,
        dts,
        rate=rate,
        use_cm=use_cm,
        ref=ref,
        window=window,
        cc_thresh=cc_thresh,
        avg_cc_thresh=avg_cc_thresh,
        sigma_filter=sigma_filter,
    )

    if outname:
        import h5py

        with h5py.File(outname, "w") as f:
            f["stackavg"] = cur_phase_sum
        sario.save_dem_to_h5(outname, sario.load("dem.rsc"))
    return cur_phase_sum, cc_stack


def select_igrams(geolist, intlist, event_date, num_igrams=None):

    insert_idx = np.searchsorted(geolist, event_date)
    num_igrams = num_igrams or len(geolist) - insert_idx

    # Since `event_date` will fit in the sorted array at `insert_idx`, then
    # geolist[insert_idx] is the first date AFTER the event
    geo_subset = geolist[insert_idx - num_igrams : insert_idx + num_igrams]

    stack_igrams = list(zip(geo_subset[:num_igrams], geo_subset[num_igrams:]))
    return stack_igrams


def select_pre_event(geolist, intlist, event_date, min_date=None):
    ifgs = [ifg for ifg in intlist if (ifg[0] < event_date and ifg[1] < event_date)]
    return _filter_min_max_date(ifgs, min_date, None)


def select_post_event(geolist, intlist, event_date, max_date=None):
    ifgs = [ifg for ifg in intlist if (ifg[0] > event_date and ifg[1] > event_date)]
    return _filter_min_max_date(ifgs, None, max_date)


def _filter_min_max_date(ifgs, min_date, max_date):
    if min_date:
        ifgs = [ifg for ifg in ifgs if (ifg[0] > min_date and ifg[1] > min_date)]
    if max_date:
        ifgs = [ifg for ifg in ifgs if (ifg[0] < max_date and ifg[1] < max_date)]
    return ifgs


def create_stack(
    stack_igrams,
    stack_fnames,
    dts,
    rate=False,
    use_cm=True,
    ref=(5, 5),
    window=5,
    cc_thresh=None,
    avg_cc_thresh=0.35,
    sigma_filter=0.3,
):
    cur_phase_sum = np.zeros(sario.load(stack_fnames[0]).shape).astype(float)
    cc_stack = np.zeros_like(cur_phase_sum)
    # for pixels that get masked sometimes, lower that count in the final stack dividing
    pixel_count = np.zeros_like(cur_phase_sum, dtype=int)
    dt_total = 0
    for f, dt in zip(stack_fnames, dts):
        deramped_phase = remove_ramp(sario.load(f), deramp_order=1, mask=np.ma.nomask)
        cur_cc = sario.load(f.replace(".unw", ".cc"))

        if cc_thresh:
            bad_pixel_mask = cur_cc < cc_thresh
        else:
            # zeros => dont mask any to nan
            bad_pixel_mask = np.zeros_like(deramped_phase, dtype=bool)

        deramped_phase[bad_pixel_mask] = np.nan

        # cur_phase_sum += deramped_phase
        cur_phase_sum = np.nansum(np.stack([cur_phase_sum, deramped_phase]), axis=0)
        pixel_count += (~bad_pixel_mask).astype(int)
        dt_total += (~bad_pixel_mask) * dt

        cc_stack += cur_cc

    # subtract the reference location:
    ref_row, ref_col = ref
    win = window // 2
    patch = cur_phase_sum[
        ref_row - win : ref_row + win + 1, ref_col - win : ref_col + win + 1
    ]
    cur_phase_sum -= np.nanmean(patch)

    if rate:
        cur_phase_sum /= dt_total
    else:
        cur_phase_sum /= pixel_count
    cc_stack /= len(stack_fnames)

    if avg_cc_thresh:
        cur_phase_sum[cc_stack < avg_cc_thresh] = np.nan

    if use_cm:
        cur_phase_sum *= PHASE_TO_CM

    if sigma_filter:
        import insar.blob.utils as blob_utils

        cur_phase_sum = blob_utils.gaussian_filter_nan(cur_phase_sum, sigma_filter)

    return cur_phase_sum, cc_stack


def subset_stack(
    point,
    event_date,
    ref=(3, 3),
    window=3,
    nigrams=10,
    ignore_geos=True,
    min_date=None,
    max_date=None,
):
    gi_file = "geolist_ignore.txt" if ignore_geos else None
    geolist, intlist = sario.load_geolist_intlist(".", geolist_ignore_file=gi_file)

    # stack_igrams = select_igrams(geolist, intlist, event_date, nigrams)
    # stack_igrams = select_pre_event(geolist, intlist, event_date, min_date=date(2019, 7, 1))
    stack_igrams = select_post_event(
        geolist, intlist, event_date, max_date=date(2020, 5, 1)
    )

    stack_fnames = sario.intlist_to_filenames(stack_igrams, ".unw")
    # dts = [(pair[1] - pair[0]).days for pair in stack_igrams]
    phase_subset_stack = []
    for f in stack_fnames:
        cur = subset.read_subset(
            subset.bbox_around_point(*point), f, driver="ROI_PAC", bands=[2]
        )
        deramped_phase = remove_ramp(np.squeeze(cur), deramp_order=1, mask=np.ma.nomask)
        phase_subset_stack.append(deramped_phase)

    phase_subset_stack = np.mean(np.stack(phase_subset_stack, axis=0), axis=0)
    # subtract the reference location:
    ref_row, ref_col = ref
    win = window // 2
    patch = phase_subset_stack[
        ref_row - win : ref_row + win + 1, ref_col - win : ref_col + win + 1
    ]
    phase_subset_stack -= np.nanmean(patch)
    phase_subset_stack *= PHASE_TO_CM
    return phase_subset_stack
