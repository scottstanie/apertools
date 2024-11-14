#! /usr/bin/env python
"""Author: Scott Staniewicz
utils.py: Miscellaneous helper functions
Email: scott.stanie@utexas.edu
"""
from __future__ import division, print_function
import contextlib
import datetime
import copy
import errno
import sys
import psutil
import os
import subprocess
import numpy as np
import inspect
import itertools
from pathlib import Path

from apertools.log import get_log

logger = get_log()


def get_file_ext(filename):
    """Extracts the file extension, including the '.' (e.g.: .slc)

    Examples:
        >>> print(get_file_ext('radarimage.slc'))
        .slc
        >>> print(get_file_ext('unwrapped.lowpass.unw'))
        .unw

    """
    return os.path.splitext(filename)[1]


def rewrap_to_2pi(phase, n=2):
    """Converts phase results to be centered from -pi to pi

    The result from calculating, e.g., closure phase, will usually
    have many values centered around -2pi, 0, and 2pi.
    This function puts them all centered around 0.

    Args:
        phase (ndarray): array (or scalar) of phase values

    Returns:
        re-wrapped values within the interval -pi to pi
    """
    return np.mod(phase + np.pi * n / 2, n * np.pi) - (np.pi * n / 2)


def round_to(x, step):
    """Round `x` to the nearest `step`."""
    return step * np.round(x / step)


def to_datetime(dates, tzinfo=datetime.timezone.utc):
    """Convert a single (or list of) `datetime.date` to timezone-aware `datetime.datetime`"""
    if isinstance(dates, datetime.datetime):
        return datetime.datetime(*dates.timetuple()[:6], tzinfo=tzinfo)
    try:
        iter(dates)
        if len(dates) == 0:
            return dates
        try:  # Check if its a list of tuples (an ifglist)
            iter(dates[0])
            return [to_datetime(tup) for tup in dates]
        except TypeError:
            return [datetime.datetime(*d.timetuple()[:6], tzinfo=tzinfo) for d in dates]
    # Or if it's just one sigle date
    except TypeError:
        return datetime.datetime(*dates.timetuple()[:6], tzinfo=tzinfo)


def get_open_handles(file_pattern=""):
    out = []
    for proc in psutil.process_iter():
        try:
            ofs = proc.open_files()
            if len(ofs) == 0:
                continue
            if any(file_pattern in pp.path for pp in ofs):
                out.append(proc, ofs)
        except Exception:
            continue
    return out


def variable_sizes(n=20, min_size=1):
    """Returns sizes of top n local variables larger than min_size MB"""

    # Get the variables one scope up (whatever is calling this function)
    def sizeof_fmt(num, suffix="B"):
        # https://stackoverflow.com/a/1094933/1870254
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, "Yi", suffix)

    outer_vars = inspect.currentframe().f_back.f_locals
    out_str = ""
    for size, name in sorted(
        ((sys.getsizeof(value), name) for name, value in outer_vars.items()),
        reverse=True,
    )[:n]:
        if size > min_size * 1024 * 1024:
            out_str += f"{name:>30}: {sizeof_fmt(size):>8}\n"
    return out_str


# NOTE: now possible in numpy 1.20:
# def take_looks(x, rl, cl, func=np.mean):
#    from numpy.lib.stride_tricks import sliding_window_view
#    views = sliding_window_view(x, (rl, cl))
#    return func(views[::rl, ::cl], axis=(2, 3))


def take_looks(arr, row_looks, col_looks, separate_complex=False, **kwargs):
    """Downsample a numpy matrix by summing blocks of (row_looks, col_looks)

    Cuts off values if the size isn't divisible by num looks

    NOTE: For complex data, looks on the magnitude are done separately
    from looks on the phase

    Args:
        arr (ndarray) 2D array of an image
        row_looks (int) the reduction rate in row direction
        col_looks (int) the reduction rate in col direction
        separate_complex (bool): take looks on magnitude and phase separately
            Better to preserve the look of the magnitude

    Returns:
        ndarray, size = ceil(rows / row_looks, cols / col_looks)
    """
    if row_looks == 1 and col_looks == 1:
        return arr
    if isinstance(arr, dict):
        return take_looks_rsc(arr, row_looks, col_looks)
    if arr.ndim >= 3:
        return np.stack([
            take_looks(
                a, row_looks, col_looks, separate_complex=separate_complex, **kwargs
            )
            for a in arr
        ])
    if np.iscomplexobj(arr) and separate_complex:
        mag_looked = take_looks(np.abs(arr), row_looks, col_looks)
        phase_looked = take_looks(np.angle(arr), row_looks, col_looks)
        return mag_looked * np.exp(1j * phase_looked)

    rows, cols = arr.shape
    new_rows = rows // row_looks
    new_cols = cols // col_looks

    row_cutoff = rows % row_looks
    col_cutoff = cols % col_looks

    if row_cutoff != 0:
        arr = arr[:-row_cutoff, :]
    if col_cutoff != 0:
        arr = arr[:, :-col_cutoff]
    # For taking the mean, treat integers as floats
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype("float")

    return np.mean(
        np.reshape(arr, (new_rows, row_looks, new_cols, col_looks)), axis=(3, 1)
    )


def take_looks_rsc(rsc_data, row_looks, col_looks):
    nrows, ncols = rsc_data["file_length"], rsc_data["width"]

    out_rsc = rsc_data.copy()
    out_rsc["x_step"] = rsc_data["x_step"] * col_looks
    out_rsc["y_step"] = rsc_data["y_step"] * row_looks
    out_rsc["file_length"] = nrows // row_looks
    out_rsc["width"] = ncols // col_looks
    return out_rsc


def take_looks_bn(
    arr, row_looks, col_looks, row_stride=None, col_stride=None, func_type="mean"
):
    """Multi-look window with different step sizes than look sizes."""
    import bottleneck as bn

    if row_stride is None:
        row_stride = row_looks
    if col_stride is None:
        col_stride = col_looks

    func_name = func_type.replace("nan", "")  # bottleneck always ignores nans
    if func_name not in ("sum", "mean", "median", "max"):
        raise ValueError(f"func_type {func_type} not supported")

    func = getattr(bn, f"move_{func_name}")

    # Pad at the end so we can get a centered mean
    r_pad = row_looks - row_looks // 2 - 1
    c_pad = col_looks - col_looks // 2 - 1
    # dont pad the earlier dimensions, just the last two we'll be multilooking
    pad_widths = (arr.ndim - 2) * ((0, 0),) + ((0, r_pad), (0, c_pad))
    arr0 = np.pad(arr, pad_width=pad_widths, mode="constant", constant_values=np.nan)
    # bottleneck doesn't support multi-axis, so we have to do it in two steps:
    # across cols
    a1 = func(arr0, col_looks, axis=-1, min_count=1)[..., :, c_pad:]
    # then rows
    a2 = func(a1, row_looks, axis=-2, min_count=1)[..., r_pad:, :]
    # note: if there are less than min_count non-nan values in the window, returns nan

    # if we dont pad:
    # return a2[..., (row_stride - 1) :: row_stride, (col_stride - 1) :: col_stride]
    r_start = row_stride // 2
    c_start = col_stride // 2
    return a2[..., r_start::row_stride, c_start::col_stride]


def moving_window_mean(image, size):
    """Calculate the mean of a moving window of size `size`

    Parameters
    ----------
    image : ndarray
        input image
    size : int or tuple of int
        Window size. If a single int, the window is square.
        If a tuple of (row_size, col_size), the window can be rectangular.

    Returns
    -------
    ndarray
        image the same size as `image`, where each pixel is the mean
        of the corresponding window.
    """
    if isinstance(size, int):
        size = (size, size)
    if len(size) != 2:
        raise ValueError("size must be a single int or a tuple of 2 ints")
    if size[0] % 2 == 0 or size[1] % 2 == 0:
        raise ValueError("size must be odd in both dimensions")

    row_size, col_size = size
    # Pad the image with zeros
    image = np.pad(
        image,
        ((row_size // 2, row_size // 2), (col_size // 2, col_size // 2)),
        mode="constant",
        constant_values=np.nan,
    )

    # Calculate the cumulative sum of the image in 2 dimensions
    cumsum = np.nancumsum(np.nancumsum(image, axis=0), axis=1)
    win_size = size[0] * size[1]

    # Use the integral image to calculate the moving window average
    return (
        cumsum[size[0] :, size[1] :]
        + cumsum[: -size[0], : -size[1]]
        - cumsum[size[0] :, : -size[1]]
        - cumsum[: -size[0], size[1] :]
    ) / win_size


def moving_window_std(image, size):
    """Calculate the standard deviation of a moving window of size `size`

    Parameters
    ----------
    image : ndarray
        input image
    size : int or tuple of int
        Window size. If a single int, the window is square. If a tuple,
        the window is rectangular.

    Returns
    -------
    ndarray
        image the same size as `image`, where each pixel is the standard
        deviation of the corresponding window.

    Source
    ------
    https://stackoverflow.com/a/25912169/4174466
    """
    from scipy.signal import convolve2d

    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)

    kernel = np.ones((size, size))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")
    return np.sqrt((s2 - (s**2 / ns)) / ns)


def get_looks_rdr(filename: str):
    """Get the row/col looks of a radar coordinates file from the transform"""
    import rasterio as rio

    with rio.open(filename) as src:
        x_step, _, _, _, y_step, _ = tuple(src.transform)[:6]
        # x_step is column looks, y_step is row looks
        return y_step, x_step


def scale_dset(filename, dset, scale):
    import h5py

    with h5py.File(filename, "r+") as f:
        data = f[dset]
        data[...] *= scale


def slclist_from_igrams(igram_list):
    """Takes a list of [(reference, secondary),...] igram date pairs
    and returns the list of unique dates of SAR images used to form them
    """
    return sorted(list(set(itertools.chain(*igram_list))))


def full_igram_list(slclist):
    """Create the list of all possible igram pairs from slclist"""
    return [
        (early, late)
        for (idx, early) in enumerate(slclist[:-1])
        for late in slclist[idx + 1 :]
    ]


def filter_min_max_date(ifg_list, min_date=None, max_date=None, verbose=False):
    """Filters from an iterable of (date1, date1) ifg pairs by min/max date"""
    # Coerce all dates to datetimes
    if min_date:
        md = to_datetime(min_date)
        len_before = len(ifg_list)
        ifg_list = [ifg for ifg in ifg_list if (ifg[0] >= md and ifg[1] >= md)]
        if verbose:
            logger.info(
                f"Ignoring {len_before - len(ifg_list)} igrams before {min_date}"
            )
    if max_date:
        md = to_datetime(max_date)
        len_before = len(ifg_list)
        ifg_list = [ifg for ifg in ifg_list if (ifg[0] <= md and ifg[1] <= md)]
        if verbose:
            logger.info(
                f"Ignoring {len_before - len(ifg_list)} igrams after {max_date}"
            )
    return ifg_list


def filter_slclist_ifglist(
    ifg_date_list,
    min_date=None,
    max_date=None,
    min_temporal_baseline=None,
    max_temporal_baseline=None,
    min_bandwidth=None,
    max_bandwidth=None,
    include_annual=False,
    slclist_ignore_file=None,
    verbose=False,
):
    # Make sure it's a list of tuples
    ifg_date_list = [tuple(ifg) for ifg in ifg_date_list]
    # Dont alter the original so we can find the indices at the end
    valid_ifg_pairs = copy.copy(ifg_date_list)
    if slclist_ignore_file is not None:
        _, valid_ifg_pairs = ignore_slc_dates(
            ifg_date_list=valid_ifg_pairs, slclist_ignore_file=slclist_ignore_file
        )

    valid_ifg_pairs = filter_min_max_date(
        valid_ifg_pairs, min_date, max_date, verbose=verbose
    )
    # Check for 1 year interferograms before cutting down
    if include_annual:
        annual_ifgs = find_annual_ifgs(valid_ifg_pairs)
    else:
        annual_ifgs = []

    # Now filter the rest by temp baseline or by "bandwidth" aka index distance
    if max_temporal_baseline is not None and max_bandwidth is not None:
        raise ValueError("Only can filter by one of bandwidth or temp. baseline")
    if max_temporal_baseline is not None or min_temporal_baseline is not None:
        max_temporal_baseline = max_temporal_baseline or 10000
        min_temporal_baseline = min_temporal_baseline or 0
        ll = len(valid_ifg_pairs)
        valid_ifg_pairs = [
            ifg
            for ifg in valid_ifg_pairs
            if min_temporal_baseline
            <= abs((ifg[1] - ifg[0]).days)
            <= max_temporal_baseline
        ]
        if verbose:
            logger.info(
                f"Ignoring {ll - len(valid_ifg_pairs)} longer than"
                f" {max_temporal_baseline}"
            )
    elif max_bandwidth is not None or min_bandwidth is not None:
        valid_ifg_pairs = limit_ifg_bandwidth(
            valid_ifg_pairs, max_bandwidth=max_bandwidth, min_bandwidth=min_bandwidth
        )
    valid_ifg_pairs = list(sorted(valid_ifg_pairs + annual_ifgs))

    if verbose:
        logger.info(
            f"Ignoring {len(ifg_date_list) - len(valid_ifg_pairs)} igrams total"
        )

    # Now just use the ones remaining to reform the unique SAR dates
    valid_sar_dates = list(sorted(set(itertools.chain.from_iterable(valid_ifg_pairs))))
    # Does the same as np.searchsorted, but works on a list[tuple]
    valid_ifg_idxs = [ifg_date_list.index(tup) for tup in valid_ifg_pairs]
    return valid_sar_dates, valid_ifg_pairs, valid_ifg_idxs


def ignore_slc_dates(
    slc_date_list=[],
    ifg_date_list=[],
    slclist_ignore_file="slclist_ignore.txt",
    parse=True,
):
    """Read extra file to ignore certain dates of interferograms"""
    import apertools.sario

    ignore_slcs = set(
        to_datetime(
            apertools.sario.find_slcs(filename=slclist_ignore_file, parse=parse)
        )
    )
    logger.info("Ignoring the following slc dates:")
    logger.info(sorted(ignore_slcs))
    valid_slcs = [g for g in slc_date_list if g not in ignore_slcs]
    valid_igrams = [
        i for i in ifg_date_list if i[0] not in ignore_slcs and i[1] not in ignore_slcs
    ]
    logger.info(
        f"Ignoring {len(ifg_date_list) - len(valid_igrams)} igrams listed in"
        f" {slclist_ignore_file}"
    )
    return valid_slcs, valid_igrams


def limit_ifg_bandwidth(valid_ifg_dates, max_bandwidth=None, min_bandwidth=None):
    """Limit the total interferograms to just nearest `max_bandwidth` connections
    Alternative to a temportal baseline.
    """
    if not max_bandwidth:
        max_bandwidth = np.inf
    if not min_bandwidth:
        min_bandwidth = 1
    cur_early = None
    cur_bw = 1
    ifg_used = []
    for early, late in valid_ifg_dates:
        if early != cur_early:
            cur_bw = 1
            cur_early = early
        else:
            cur_bw += 1
        if cur_bw <= max_bandwidth and cur_bw >= min_bandwidth:
            ifg_used.append((early, late))
    return ifg_used


def _temp_baseline(ifg_pair):
    return (ifg_pair[1] - ifg_pair[0]).days


def find_annual_ifgs(ifg_pairs, buffer_days=30, num_years=1):
    """Pick out interferograms which are closest to 1 year in span"""
    # We only want to pick 1 ifg per date, closest to a year, but skip a date if it
    # doesn't have an ifg of baseline 365 +/- `buffer_days`
    # sar_dates = list(sorted(set(itertools.chain.from_iterable(ifg_pairs))))
    date_to_ifg = {}
    # Used to keep track how far into ifg_list the last sar date was (avoid iterations)
    for ifg in ifg_pairs:
        early = ifg[0]
        tb = _temp_baseline(ifg)
        if abs(tb - 365) > buffer_days:
            continue
        cur_ifg = date_to_ifg.get(early)
        # Use this ifg as the annual if none exist, or if it's closer to 365
        if cur_ifg is None or abs(tb - 365) < _temp_baseline(cur_ifg):
            date_to_ifg[early] = ifg
    return list(sorted(date_to_ifg.values()))


def take_looks_gdal(outname, src_filename, row_looks, col_looks, format="ROI_PAC"):
    """Downsample an array on disk using gdal_translate

    Cuts off values if the size isn't divisible by num looks

    NOTE: For complex data, looks on the magnitude are done separately
    from looks on the phase

    See https://github.com/OSGeo/gdal/blob/master/swig/python/osgeo/gdal.py#L328
    for options

    Args:
        outname (string): output/destination filename
        filename (string) Name of gdal-compatible input raster file
        row_looks (int) the reduction rate in row direction
        col_looks (int) the reduction rate in col direction

    Returns:
        ndarray, size = ceil(rows / row_looks, cols / col_looks)
        values at each pixel are averaged from input array
    """
    from osgeo import gdal
    from osgeo import gdalconst

    if row_looks == 1 and col_looks == 1:
        raise ValueError("Must take looks for file on disk")
    in_ds = gdal.Open(src_filename)
    shape = (in_ds.RasterYSize, in_ds.RasterXSize)  # (rows, cols)
    new_rows, new_cols = shape[0] // row_looks, shape[1] // col_looks
    return gdal.Translate(
        outname,
        in_ds,
        height=new_rows,
        width=new_cols,
        format=format,
        resampleAlg=gdalconst.GRIORA_Average,
    )


def take_looks_gdal2(
    infile,
    outfile,
    fmt="GTiff",
    xlooks=None,
    ylooks=None,
    noData=None,
    method="average",
):
    """
    infile - Input file to multilook
    outfile - Output file to multilook
    fmt - Output format
    xlooks - Number of looks in x/range direction
    ylooks - Number of looks in y/azimuth direction
    """
    from osgeo import gdal

    ds = gdal.Open(infile, gdal.GA_ReadOnly)

    # Input file dimensions
    xSize = ds.RasterXSize
    ySize = ds.RasterYSize

    # Output file dimensions
    outXSize = xSize // xlooks
    outYSize = ySize // ylooks

    # Set up options for translation
    gdalTranslateOpts = gdal.TranslateOptions(
        format=fmt,
        width=outXSize,
        height=outYSize,
        srcWin=[0, 0, outXSize * xlooks, outYSize * ylooks],
        noData=noData,
        resampleAlg=method,
    )

    # Call gdal_translate
    gdal.Translate(outfile, ds, options=gdalTranslateOpts)
    ds = None


def crossmul_gdal(outfile, file1, file2, row_looks, col_looks, format="ROI_PAC"):
    """Uses gdal_calc.py to multiply, then gdal_translate for looks"""
    tmp = "tmp.tif"
    cmd = """gdal_calc.py -A {f1} -B {f1} --outfile={tmp} \
            --calc="A * conj(B)" --NoDataValue=0 """.format(f1=file1, f2=file2, tmp=tmp)
    subprocess.check_call(cmd, shell=True)
    take_looks_gdal(outfile, tmp, row_looks, col_looks, format=format)
    os.remove(tmp)


def calc_igram_cor(slc1, slc2, *, row_looks=1, col_looks=1, deramp=False):
    """Cross multiply two SLCs to compute an interferogram and correlation

    The single looked version is ``I = slc1 * slc2.conj()``

    Parameters
    ----------
    slc1 : ndarray
        Reference SLC
    slc2 : ndarray
        Secondary SLC
    row_looks : int, optional
        Number of looks to take along row axis, by default 1
    col_looks : int, optional
        Number of looks to take along column axis, by default 1

    Returns
    -------
    igram : ndarray
        Interferogram image
    cor : ndarray
        Correlation image
    """

    def abs2(x):
        return x.real**2 + x.imag**2

    ifg_full = slc1 * np.conj(slc2)
    if deramp:
        ifg_full = deramp_wrapped_interferogram(ifg_full)
    ifg = take_looks(ifg_full, row_looks, col_looks)
    amp_slc1 = np.sqrt(take_looks(abs2(slc1), row_looks, col_looks))
    amp_slc2 = np.sqrt(take_looks(abs2(slc2), row_looks, col_looks))
    amp_ifg = np.abs(ifg)
    cor = np.real(amp_ifg / (amp_slc1 * amp_slc2 + 1e-9))
    return ifg, cor


def db(image):
    """Converts magnitude amplitude image to log scale"""
    if np.iscomplexobj(image):
        image = np.abs(image)
    return 20 * np.log10(image)


def mag(db_image):
    """Reverse of log/db: decibel to magnitude"""
    return 10 ** (db_image / 20)


def mask_zeros(image):
    """Turn image into masked array, 0s masked"""
    return np.ma.masked_equal(image, 0)


def force_column(arr):
    """Turns 1d numpy array into an (N, 1) shaped column"""
    return arr.reshape((len(arr), 1))


def atleast_2d(*arys):
    """column version of numpy's atleast_2d

    Reshapes to be (N, 1) if 1d
    """
    res = []
    for ary in arys:
        ary = np.asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1)
        elif ary.ndim == 1:
            result = ary[:, np.newaxis]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


def percent_zero(arr=None):
    """Function to give the percentage of a file that is exactly zero

    Used as a quality assessment check

    Args:
        arr (ndarray): pre-loaded array to check

    Returns:
        float: decimal from 0 to 1, ratio of zeros to total entries

    Example:
        >>> a = np.array([[1 + 1j, 0.0], [1, 0.0001]])
        >>> print(percent_zero(arr=a))
        0.25
    """
    return np.sum(arr == 0) / arr.size


def sliding_window_view(x, shape, step=None):
    """
    Create sliding window views of the N dimensions array with the given window
    shape. Window slides across each dimension of `x` and provides subsets of `x`
    at any window position.

    Adapted from https://github.com/numpy/numpy/pull/10771

    Args:
        x (ndarray): Array to create sliding window views.
        shape (sequence of int): The shape of the window.
            Must have same length as number of input array dimensions.
        step: (sequence of int), optional
            The steps of window shifts for each dimension on input array at a time.
            If given, must have same length as number of input array dimensions.
            Defaults to 1 on all dimensions.
    Returns:
        ndarray: Sliding window views (or copies) of `x`.
            view.shape = (x.shape - shape) // step + 1

    Notes
    -----
    ``sliding_window_view`` create sliding window views of the N dimensions array
    with the given window shape and its implementation based on ``as_strided``.
    The returned views are *readonly* due to the numpy sliding tricks.
    Examples
    --------
    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> sliding_window_view(x, shape)[0, 0]
    array([[ 0,  1],
           [10, 11]])
    >>> sliding_window_view(x, shape)[1, 2]
    array([[12, 13],
           [22, 23]])
    """
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False)

    try:
        shape = np.array(shape, np.int)
    except ValueError:
        raise TypeError("`shape` must be a sequence of integer")
    else:
        if shape.ndim > 1:
            raise ValueError("`shape` must be one-dimensional sequence of integer")
        if len(x.shape) != len(shape):
            raise ValueError("`shape` length doesn't match with input array dimensions")
        if np.any(shape <= 0):
            raise ValueError("`shape` cannot contain non-positive value")

    if step is None:
        step = np.ones(len(x.shape), np.intp)
    else:
        try:
            step = np.array(step, np.intp)
        except ValueError:
            raise TypeError("`step` must be a sequence of integer")
        else:
            if step.ndim > 1:
                raise ValueError("`step` must be one-dimensional sequence of integer")
            if len(x.shape) != len(step):
                raise ValueError(
                    "`step` length doesn't match with input array dimensions"
                )
            if np.any(step <= 0):
                raise ValueError("`step` cannot contain non-positive value")

    o = (np.array(x.shape) - shape) // step + 1  # output shape
    if np.any(o <= 0):
        raise ValueError("window shape cannot larger than input array shape")

    strides = x.strides
    view_strides = strides * step

    view_shape = np.concatenate((o, shape), axis=0)
    view_strides = np.concatenate((view_strides, strides), axis=0)
    view = np.lib.stride_tricks.as_strided(x, view_shape, view_strides, writeable=False)

    return view


def window_stack(stack, row, col, window_size=3, func=np.mean):
    """Combines square around (row, col) in 3D stack to a 1D array

    Used to average around a pixel in a stack and produce a timeseries

    Args:
        stack (ndarray): 3D array of images, stacked along axis=0
        row (int): row index of the reference pixel to subtract
        col (int): col index of the reference pixel to subtract
        window_size (int): size of the group around ref pixel to avg for reference.
            if window_size=1 or None, only the single pixel location used for output
        func (str): default=np.mean, numpy function to use on window.

    Raises:
        ValueError: if window_size is not a positive int, or if ref pixel out of bounds
    """
    if func == "identity":

        def func(arr, axis=None):
            return arr

    window_size = window_size or 1
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError(
            "Invalid window_size %s: must be odd positive int" % window_size
        )
    elif row > stack.shape[-2] or col > stack.shape[-1]:
        raise ValueError(
            "(%s, %s) out of bounds reference for stack size %s"
            % (row, col, stack.shape)
        )

    if window_size % 2 == 0:
        window_size -= 1
        print("Making window_size an odd number (%s) to get square" % window_size)

    half_win = window_size // 2
    row_slice = slice(row - half_win, row + half_win + 1)
    col_slice = slice(col - half_win, col + half_win + 1)
    subset = stack[..., row_slice, col_slice]
    return func(subset, axis=(-2, -1))


def window_stack_xr(
    da,
    lon=None,
    lat=None,
    row=None,
    col=None,
    window_size=5,
    lon_name="lon",
    lat_name="lat",
    func=np.mean,
):
    """Combines square around (row, col) in 3D DataArray to a 1D array

    Used to average around a pixel in a stack and produce a timeseries
    """
    is_2d_latlon = getattr(da, lon_name).ndim
    if row is None or col is None:
        if is_2d_latlon == 2:
            # It wont be in the indices in this case, just a 2d data array
            from apertools import latlon

            row, col = latlon.latlon_to_rowcol_rdr(
                lat,
                lon,
                lat_arr=getattr(da, lat_name).data,
                lon_arr=getattr(da, lon_name).data,
            )
            if col is None or row is None:
                # out of bounds (could be on a diagonal corner of the bbox)
                raise ValueError(f"({lon}, {lat}) is out of bounds for {da}")
        else:
            col = da.indexes[lon_name].get_indexer([lon], method="nearest")[0]
            row = da.indexes[lat_name].get_indexer([lat], method="nearest")[0]

    return window_stack(da, row, col, window_size=window_size, func=func)


def interpolate_xr(
    ds, dset_name=None, freq="6M", col="date", round_min=True, round_max=True
):
    """Interpolate an xarray dataset at a given frequency

    Args:
        ds (xr.Dataset): input dataset
        dset_name (str, optional): Name of DataArray within ds to interpolate.
        freq (str, optional): Pandas timeseries frequency. Defaults to "6M".
        col (str, optional): column containing the time coordinate. Defaults to "date".

    Returns:
        [xr.Dataset]: interpolated dataset
    """
    import pandas as pd

    dmin = pd.to_datetime(ds[col].min().to_pandas())
    dmax = pd.to_datetime(ds[col].max().to_pandas())
    if round_min:
        dmin += pd.tseries.offsets.QuarterBegin()
    if round_max:
        dmax += pd.tseries.offsets.QuarterEnd()

    date_range = pd.date_range(dmin, dmax, freq=freq)

    d = ds[dset_name] if dset_name is not None else ds
    out = d.interp(date=date_range, kwargs={"fill_value": "extrapolate"})
    if dset_name is not None:
        out = out.to_dataset()
    return out


def stack_to_xr(
    data,
    x_coords=None,
    y_coords=None,
    z_coords=None,
    dims=["z", "y", "x"],
    name="stack",
):
    import xarray as xr

    if x_coords is None:
        x_coords = np.arange(data.shape[-1])
    if y_coords is None:
        y_coords = np.arange(data.shape[-2])
    if z_coords is None:
        z_coords = np.arange(data.shape[-3])
    return xr.DataArray(
        data=data, coords=[z_coords, y_coords, x_coords], dims=dims, name=name
    )


def get_xr_transect(da, lon, lat, lon_name="lon", lat_name="lat"):
    import geopandas as gpd
    from shapely import geometry
    import xarray as xr

    # Setup an example geodataframe:
    gdf = gpd.GeoDataFrame(
        geometry=[
            geometry.LineString([(0.0, 0.0), (5.0, 5.0)]),
            geometry.LineString([(10.0, 10.0), (15.0, 15.0)]),
        ]
    )

    # Get the centroids, and create the indexers for the DataArray:
    centroids = gdf.centroid
    x_indexer = xr.DataArray(centroids.x, dims=["point"])
    y_indexer = xr.DataArray(centroids.y, dims=["point"])

    # Grab the results:
    da.sel(x=x_indexer, y=y_indexer, method="nearest")


def shift_latlon(da, full_pixel=False, down_right=False, copy=True):
    """Shift lat/lon coordinates by one pixel

    Args:
        da (xr.DataArray): input DataArray
        full_pixel (bool, optional): Shift by 1 full pixel (as opposed to half). Defaults to False.
        down_right (bool, optional): Shift latlon to southwest. Defaults to False.
        copy (bool, optional): Return a copy of `da`. Defaults to True.

    Returns:
        xr.DataArray: Shifted version of `da`
    """

    denom = 1 if full_pixel else 2
    if down_right:
        denom *= -1

    dlat = np.diff(da.lat)[0]
    dlon = np.diff(da.lon)[0]
    lats = da.lat.copy()
    lons = da.lon.copy()
    lats = lats - dlat / denom
    lons = lons - dlon / denom

    out_da = da if not copy else da.copy()
    out_da["lon"] = lons
    out_da["lat"] = lats
    return out_da


# Randoms using the sentinelapi
def find_slc_products(api, gj_obj, date_start, date_end, area_relation="contains"):
    """Query for Sentinel 1 SLC products with common options

    from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
    api = api = SentinelAPI(user, pw)
    pecosgeo = geojson_to_wkt(geojson.read_json('pecosBig.geojson'))
    find_slc_products(pecosgeo, '20150101', '20171230')

    Returns:
        OrderedDict: key = '528c0630-bbbf-4a95-8415-c55aa5ce915a', the sentinel
    """
    # area_relation : 'Intersection', 'Contains', 'IsWithin'
    # contains means that the Sentinel footprint completely contains your geojson object
    return api.query(
        gj_obj,
        date=(date_start, date_end),
        platformname="Sentinel-1",
        producttype="SLC",
        area_relation=area_relation,
    )


def fullpath(path):
    """Expands ~ and returns an absolute path"""
    return os.path.abspath(os.path.expanduser(path))


def force_symlink(src, dest):
    """python equivalent to 'ln -f -s': force overwrite"""
    if os.path.exists(fullpath(dest)) and os.path.islink(fullpath(dest)):
        os.remove(fullpath(dest))

    try:
        os.symlink(fullpath(src), fullpath(dest))
    except OSError as e:
        if e.errno == errno.EEXIST:
            raise ValueError("Non-symlink file exists: %s" % fullpath(dest))
        else:
            raise
            # os.remove(fullpath(dest))
            # os.symlink(fullpath(src), fullpath(dest))


def record_params_as_yaml(paramfile, verbose=True, **kwargs):
    from ruamel.yaml import YAML

    if verbose:
        from pprint import pprint

        pprint(kwargs)

    yaml = YAML()

    with open(paramfile, "w") as f:
        yaml.dump(kwargs, f)


def record_params_as_toml(paramfile, verbose=True, **kwargs):
    import toml

    if verbose:
        from pprint import pprint

        pprint(kwargs)

    with open(paramfile, "w") as f:
        toml.dump(kwargs, f)


def rm_if_exists(filename):
    # os.path.islink(path)  # Test for symlink
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise if different error


def get_parent_dir(filepath):
    """Shortcut to get the parent directory

    Works on directories or files, so
    "/home/scott/" -> "/home"
    "/home/scott/file.txt" -> "/home"

    Used, e.g. for when .geos should be 1 directory up from igrams

    Examples:
        >>> import tempfile, sys
        >>> if sys.version_info[0] == 2: from backports import tempfile
        >>> temp_dir = tempfile.TemporaryDirectory()
        >>> nested = os.path.join(temp_dir.name, 'dir2')
        >>> os.mkdir(nested)
        >>> get_parent_dir(nested) == temp_dir.name
        True
        >>> open(os.path.join(nested, "emptyfile"), "a").close()
        >>> get_parent_dir(os.path.join(nested, "emptyfile")) == temp_dir.name
        True
    """
    if os.path.isdir(filepath):
        return os.path.dirname(os.path.abspath(os.path.normpath(filepath)))
    else:
        return os.path.dirname(os.path.split(os.path.abspath(filepath))[0])


def get_cache_dir(force_posix=False, app_name="apertools") -> Path:
    """Returns the config folder for the application.  The default behavior
    is to return whatever is most appropriate for the operating system.

    This is used to store gps timeseries data

    the following folders could be returned:
    Mac OS X:
      ``~/Library/Application Support/apertools``
    Mac OS X (POSIX):
      ``~/.apertools``
    Unix:
      ``~/.cache/apertools``
    Unix (POSIX):
      ``~/.apertools``

    Args:
        force_posix: if this is set to `True` then on any POSIX system the
            folder will be stored in the home folder with a leading
            dot instead of the XDG config home or darwin's
            application support folder.

    Source: https://github.com/pallets/click/blob/master/click/utils.py#L368
    """

    if force_posix:
        path = os.path.join(os.path.expanduser("~/." + app_name))
    if sys.platform == "darwin":
        path = os.path.join(
            os.path.expanduser("~/Library/Application Support"), app_name
        )
    path = os.path.join(
        os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.cache")),
        app_name,
    )
    if not os.path.exists(path):
        os.makedirs(path)
    return Path(path)


def az_inc_to_enu(
    infile="los.rdr",
    outfile="los_enu.tif",
    do_geocode=True,
    latlon_step=0.001,
    invert=True,
    nodata=0,
):
    """Convert a 2 band azimuth/inclindation line of sight file to 3 band east/north/up

    Source: http://earthdef.caltech.edu/boards/4/topics/327
    Args:
        infile (str, optional): 2-band los file. Defaults to "los.rdr".
        outfile (str, optional): output ENU file. Defaults to "los_enu.tif".
        do_geocode (bool, optional): If the `infile` is in radar coordinates (e.g.
            ISCE's geometry folder has "los.rdr"). Defaults to True.
        latlon_step (float, optional): if geocoding, lat/lon spacing. Defaults to 0.001.
        invert (bool, optional): Reverse the LOS convention in the output. Defaults to True.
            E.g. ISCE uses "from ground toward satellite" line of sight vector convention,
            so the "up" component is positive".
            `invert=True` makes the output use "satellite-to-ground" convention, and "up" is
            negative.
        nodata (float): value to use for nodata if geocoding. Defaults to 0.
    """
    cmd = (
        f"gdal_calc.py --quiet -A {infile} -B {infile} --A_band=1 --B_band=2 --outfile"
        ' tmp_los_east.tif --calc="sin(deg2rad(A)) * cos(deg2rad(B+90))" '
    )
    print(cmd)
    subprocess.run(cmd, check=True, shell=True)

    cmd = (
        f"gdal_calc.py --quiet -A {infile} -B {infile} --A_band=1 --B_band=2 --outfile"
        ' tmp_los_north.tif --calc="sin(deg2rad(A)) * sin(deg2rad(B+90))" '
    )
    print(cmd)
    subprocess.run(cmd, check=True, shell=True)

    cmd = (
        f"gdal_calc.py --quiet -A {infile} -B {infile} --A_band=1 --B_band=2 --outfile"
        ' tmp_los_up.tif --calc="cos(deg2rad(A))" '
    )
    print(cmd)
    subprocess.run(cmd, check=True, shell=True)

    cmd = (
        f"gdal_merge.py -separate -o {outfile} tmp_los_east.tif tmp_los_north.tif"
        " tmp_los_up.tif "
    )
    print(cmd)
    subprocess.run(cmd, check=True, shell=True)
    subprocess.run(
        "rm -f tmp_los_east.tif tmp_los_north.tif tmp_los_up.tif",
        shell=True,
        check=True,
    )
    temp_enu = "temp_los_enu.tif"
    if do_geocode:
        from apertools import geocode

        logger.info("Geocoding LOS file")
        os.rename(outfile, temp_enu)
        # Assume the infile is in a geom dir...
        geom_dir = os.path.dirname(infile)
        lon_file = os.path.join(geom_dir, "lon.rdr")
        lat_file = os.path.join(geom_dir, "lat.rdr")
        geocode.geocode(
            infile=temp_enu,
            outfile=outfile,
            lat=lat_file,
            lon=lon_file,
            lat_step=latlon_step,
            lon_step=latlon_step,
            nodata=nodata,
        )
        os.remove(temp_enu)
    if invert:
        logger.info("Inverting LOS direction")
        cmd = (
            f"gdal_calc.py --quiet --NoDataValue={nodata} --allBands=A -A {outfile} "
            f' --outfile={temp_enu} --calc "-1 * A" '
        )
        logger.info(cmd)
        subprocess.run(cmd, check=True, shell=True)
        os.rename(temp_enu, outfile)
    return outfile


def enu_to_az_inc(infile, outfile="los_az_inc.tif"):
    """Convert 3-band ENU LOS to ISCE convention of azimuth-elevation

    Here, the output LOS points FROM ground, TO satellite (reveresed from our other convention).
    Channel 1: Incidence angle measured from vertical at target (always positive)
    Channel 2: Azimuth angle is measured from North in the anti-clockwise direction.

    References:
        http://earthdef.caltech.edu/boards/4/topics/1915?r=1919#message-1919
        http://earthdef.caltech.edu/boards/4/topics/327
    """
    # Note: A = xEast, B = yNorth, C = zUp
    # inc = atan2(sqrt(x**2 + y**2), abs(z))
    tmp_inc = "tmp_los_inc.tif"
    cmd = (
        f"gdal_calc.py --quiet -A {infile} -B {infile} -C {infile} --A_band=1"
        f' --B_band=2 --C_band=3 --outfile {tmp_inc} --calc="rad2deg(arctan2(sqrt(A **2'
        ' + B**2), abs(C)))" '
    )
    subprocess.run(cmd, check=True, shell=True)
    # -90 + rad2deg(arctan2( -yNorth, -xEast ))
    tmp_az = "tmp_los_az.tif"
    cmd = (
        f"gdal_calc.py --quiet -A {infile} -B {infile}  --A_band=1 --B_band=2  "
        f'--outfile {tmp_az} --calc=" -90 + rad2deg(arctan2(-B, -A ))" '
    )
    subprocess.run(cmd, check=True, shell=True)
    cmd = f"gdal_merge.py -separate -o {outfile} {tmp_inc} {tmp_az} "
    subprocess.run(cmd, check=True, shell=True)
    subprocess.run(f"rm -f {tmp_inc} {tmp_az}", shell=True, check=True)
    return outfile


def enu_to_az_inc_array(los_enu, to_deg=True):
    """Convert 3-band ENU LOS array to ISCE convention of azimuth-elevation

    Here, the output LOS points FROM ground, TO satellite (reveresed from our other convention).
    Channel 1: Incidence angle measured from vertical at target (always positive)
    Channel 2: Azimuth angle is measured from North in the anti-clockwise direction.
    """
    E, N, U = los_enu
    inc = np.arctan2(np.sqrt(E**2 + N**2), np.abs(U))
    if to_deg:
        inc = np.rad2deg(inc)
    az = -90 + np.rad2deg(np.arctan2(-N, -E))
    if not to_deg:
        az = np.deg2rad(az)
    return np.stack([inc, az])


def velo_to_cumulative_scale(slclist):
    ndays = (slclist[-1] - slclist[0]).days
    # input is MM/year
    # (mm/year) * (1 yr / 365 days) * (1 cm / 10 mm) * ndays => [cm]
    return ndays / 365 / 10


def find_largest_component_idxs(binimg, strel_size=2):
    from skimage.morphology import disk, closing
    from skimage import measure
    from collections import Counter

    selem = disk(strel_size)
    img = closing(binimg, selem)
    labels, num = measure.label(img, return_num=True, connectivity=2)
    counts = Counter(labels.reshape(-1))
    # Get the top label which is not 0, the background
    top2 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:2]
    fg_label, fg_count = top2[0] if top2[0][0] != 0 else top2[1]
    return labels == fg_label


def pprint_lon_lat(lon, lat, decimals=0):
    """Pretty print a lat and and lon

    Examples:
    >>> pprint_lon_lat(-104.52, 31.123)
    'N31W105'
    >>> pprint_lon_lat(-104.52, 31.123, decimals=1)
    'N31.1W104.5'
    """
    # Note: strings must be formatted twice:
    # ff = f"{{:02.1f}}" ; f"{hemi_ew}{ff}".format(32.123)
    # 'E32.1
    hemi_ns = "N" if lat >= 0 else "S"
    format_lat = "{:02." + str(decimals) + "f}"
    lat_str = f"{hemi_ns}{format_lat}".format(abs(lat))

    hemi_ew = "E" if lon >= 0 else "W"
    format_lon = "{:03." + str(decimals) + "f}"
    lon_str = f"{hemi_ew}{format_lon}".format(abs(lon))
    return f"{lat_str}{lon_str}"


def block_slices(arr_shape, block_shape, overlaps=(0, 0), start_offsets=(0, 0)):
    """Iterator to get indexes for accessing blocks of a raster

    Args:
        arr_shape = (num_rows, num_cols), full size of array to access
        block_shape = (height, width), size of accessing blocks
        overlaps = (row_overlap, col_overlap), number of pixels to re-include
            after sliding the block (default (0, 0))
        start_offset = (row_offset, col_offset) starting location (default (0,0))
    Yields:
        iterator: ((row_start, row_end), (col_start, col_end))

    Notes:
        If the block_shape/overlaps don't evenly divide the full arr_shape,
        It will return the edges as smaller blocks, rather than skip them

    Examples:
    >>> list(block_slices((180, 250), (100, 100)))
    [((0, 100), (0, 100)), ((0, 100), (100, 200)), ((0, 100), (200, 250)), \
((100, 180), (0, 100)), ((100, 180), (100, 200)), ((100, 180), (200, 250))]
    >>> list(block_slices((180, 250), (100, 100), overlaps=(10, 10)))
    [((0, 100), (0, 100)), ((0, 100), (90, 190)), ((0, 100), (180, 250)), \
((90, 180), (0, 100)), ((90, 180), (90, 190)), ((90, 180), (180, 250))]
    """
    rows, cols = arr_shape
    row_off, col_off = start_offsets
    row_overlap, col_overlap = overlaps
    height, width = block_shape

    if height is None:
        height = rows
    if width is None:
        width = cols

    # Check we're not moving backwards with the overlap:
    if row_overlap >= height:
        raise ValueError(f"{row_overlap = } must be less than {height = }")
    if col_overlap >= width:
        raise ValueError(f"{col_overlap = } must be less than {width = }")
    while row_off < rows:
        while col_off < cols:
            row_end = min(row_off + height, rows)  # Dont yield something OOB
            col_end = min(col_off + width, cols)
            yield ((row_off, row_end), (col_off, col_end))

            col_off += width
            if col_off < cols:  # dont bring back if already at edge
                col_off -= col_overlap

        row_off += height
        if row_off < rows:
            row_off -= row_overlap
        col_off = 0


def iter_blocks(
    filename, band=1, window_shape=(None, None), overlaps=(0, 0), start_offsets=(0, 0)
):
    from rasterio.windows import Window
    import rasterio as rio

    with rio.open(filename) as src:
        block_iter = block_slices(
            src.shape,
            window_shape,
            overlaps=overlaps,
            start_offsets=start_offsets,
        )
        for win_slice in block_iter:
            window = Window.from_slices(*win_slice)
            yield src.read(band, window=window)


def memmap_blocks(
    filename, full_shape, block_rows, dtype, overlaps=(0, 0), start_offsets=(0, 0)
):
    total_rows, total_cols = full_shape
    block_iter = block_slices(
        full_shape,
        (block_rows, total_cols),
        overlaps=overlaps,
        start_offsets=start_offsets,
    )
    dtype = np.dtype(dtype)
    for block_idxs in block_iter:
        row_start = block_idxs[0][0]
        # Make sure we don't ask for beyond end of file
        cur_rows = min(block_rows, total_rows - row_start)
        offset = total_cols * row_start * dtype.itemsize
        cur_block = np.memmap(
            filename,
            mode="r",
            dtype=dtype,
            offset=offset,
            shape=(cur_rows, total_cols),
        )
        yield cur_block


def get_h5stack_block_shape(
    h5_stack_file, dset, max_bytes=100e6, default_chunk_size=(None, 10, 10)
):
    """Find shape to load from `h5_stack_file` with memory size < `max_bytes`.

    Args:
        h5_stack_file (str): HDF5 file name containing 3D dataset
        dset (str): name of 3D dataset within `h5_stack_file`
        max_bytes (float, optional): target size of memory for blocks.
            Defaults to 100e6.
        default_chunk_size (tuple/list, optional): If `dset` is not chunked,
            size to use as chunks.  Defaults to (None, 10, 10).

    Returns:
        tuple[int]: (num_rows, num_cols) of block shape to load
    """
    import h5py

    with h5py.File(h5_stack_file) as hf:
        full_shape = hf[dset].shape
        nstack = full_shape[0]
        # Use the HDF5 chunk size, if the dataset is chunked.
        chunk_size = list(hf[dset].chunks) or default_chunk_size
        chunk_size[0] = nstack  # always load a full depth slice at once

        nbytes = hf[dset].dtype.itemsize

    return _get_stack_block_shape(full_shape, chunk_size, nbytes, max_bytes)


def get_gdal_block_shape(
    filename, max_bytes=100e6, default_chunk_size=(None, 128, 128)
):
    """Find shape to load from GDAL-readable `filename` with memory size < `max_bytes`.

    Args:
        vrt_file (str): VRT file name containing 3D dataset
        max_bytes (float, optional): target size of memory (in Bytes)
            for each block.
            Defaults to 100e6.
        default_chunk_size (tuple/list, optional): If `filename` is not chunked/blocked,
            size to use as chunks.  Defaults to (None, 128, 128).

    Returns:
        tuple[int]: (num_rows, num_cols) shape of blocks to load from `vrt_file`
    """
    import rasterio as rio

    with rio.open(filename) as src:
        full_shape = src.shape
        if src.block_shapes:
            chunk_size = [src.count, *src.block_shapes[0]]
        else:
            chunk_size = default_chunk_size

        nbytes = np.dtype(src.dtypes[0]).itemsize

    return _get_stack_block_shape(full_shape, chunk_size, nbytes, max_bytes)


def _get_stack_block_shape(full_shape, chunk_size, nbytes, max_bytes):
    """Figure out how much to load at 1 time, staying at ~`max_bytes` bytes of RAM"""
    chunks_per_block = max_bytes / (np.prod(chunk_size) * nbytes)
    row_chunks, col_chunks = 1, 1
    cur_block_shape = list(copy.copy(chunk_size))
    while chunks_per_block > 1:
        # First keep incrementing the number of rows we grab at once time
        if row_chunks * chunk_size[1] < full_shape[1]:
            row_chunks += 1
            cur_block_shape[1] = min(row_chunks * chunk_size[1], full_shape[1])
        # Then increase the column size if still haven't hit `max_bytes`
        elif col_chunks * chunk_size[2] < full_shape[2]:
            col_chunks += 1
            cur_block_shape[2] = min(col_chunks * chunk_size[2], full_shape[2])
        else:
            break
        chunks_per_block = max_bytes / (np.prod(cur_block_shape) * nbytes)
    return cur_block_shape


def ifg_to_mag_phase(filename, outname=None, driver=None):
    """Convert a complex float interferogram into a 2-band raster of magnitude/phase"""
    import rasterio as rio
    from copy import deepcopy

    if not outname:
        _, ext = os.path.splitext(filename)
        outname = filename.replace(ext, ext + ".mph")
        driver = "ENVI"
        print("saving to", outname, "with driver", driver)

    with rio.open(filename) as src:
        arr = src.read(1)
        out_meta = deepcopy(src.meta)
        out_meta["count"] = 2
        out_meta["dtype"] = "float32"
        allowed_drivers = ("isce", "envi", "gtiff")
        if driver:
            if driver.lower() not in allowed_drivers:
                raise ValueError("Driver must be in {}".format(allowed_drivers))
            out_meta["driver"] = driver
        if out_meta["driver"].lower() == "envi":
            # make sure the .hdr is appended after everything
            #  -of ENVI -co SUFFIX=ADD
            out_meta["SUFFIX"] = "ADD"
        print(out_meta)
        with rio.open(outname, "w", **out_meta) as dst:
            dst.write(np.abs(arr), 1)
            dst.write(np.angle(arr), 2)


@contextlib.contextmanager
def chdir_then_revert(path):
    """Temporarily change directory to `path`, then go back to original working dir

    with chdir_then_revert('temp_dir'):
        #...do stuff
    # now back in original
    """
    orig_path = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(orig_path)


def values_per_date(values, ifg_date_list, as_dataframe=False):
    """Get the values for each date in `ifg_date_list`"""
    values = np.array(values)
    sar_date_list = list(sorted(set(itertools.chain.from_iterable(ifg_date_list))))
    out_dict = {}
    for idx, cur_date in enumerate(sar_date_list):
        cur_data = [
            (idx, date_pair)
            for idx, date_pair in enumerate(ifg_date_list)
            if cur_date in date_pair
        ]
        idxs, _ = zip(*cur_data)
        out_dict[cur_date] = values[np.array(idxs)]
    if not as_dataframe:
        return out_dict
    import pandas as pd

    return pd.DataFrame(data=out_dict).melt(var_name="sar_date", value_name="value")


def get_outlier_dates(values, ifg_date_list, nsigma=4, scale=1.4826):
    df = values_per_date(values, ifg_date_list, as_dataframe=True)
    try:
        # Cant seem to groupby when it's a Timestamp
        df["sar_date"] = df["sar_date"].dt.date
    except:
        pass
    df["value"] = df["value"].abs()
    mean_abs_val = df.groupby("sar_date").mean()
    # mean_abs_val.rename({"value": "mean abs value"}, axis=1, inplace=True)
    med = mean_abs_val.median()
    cutoff = med + nsigma * scale * np.abs(mean_abs_val - med).median()
    # from scipy import stats
    # cutoff = med + nsigma * stats.median_abs_deviation(mean_abs_val, scale=scale)
    # return cutoff
    outlier_df = mean_abs_val > cutoff
    outlier_dates = outlier_df[outlier_df.value].index
    return cutoff, mean_abs_val, to_datetime(outlier_dates)


def prune_outlier_values(values, ifg_date_list, nsigma=4, scale=1.4826):
    cutoff, outlier_df, outlier_dates = get_outlier_dates(
        values, ifg_date_list, nsigma, scale
    )
    good_tuples = [
        (val, pair)
        for (val, pair) in zip(values, ifg_date_list)
        if pair[0] not in outlier_dates and pair[1] not in outlier_dates
    ]
    values_good, ifg_date_list_good = zip(*good_tuples)
    return (
        np.array(values_good),
        ifg_date_list_good,
    )  # , cutoff, outlier_df, outlier_dates


def cross_section(data, start, end, steps=100, interp_type="linear", crs="epsg:4326"):
    """Obtain an interpolated cross-sectional slice through gridded data.
    Utilizing the interpolation functionality in `xarray`, this function takes a vertical
    cross-sectional slice along a geodesic through the given data on a regular grid, which is
    given as an `xarray.DataArray` so that we can utilize its coordinate and projection
    metadata.
    Parameters
    ----------
    data: `xarray.DataArray` or `xarray.Dataset`
        Three- (or higher) dimensional field(s) to interpolate. The DataArray (or each
        DataArray in the Dataset) must have been parsed by MetPy and include both an x and
        y coordinate dimension and the added `crs` coordinate.
    start: (2, ) array_like
        A longitude-latitude designating the start point of the geodesic (units are
        degrees east and degrees north).
    end: (2, ) array_like
        A longitude-latitude pair designating the end point of the geodesic (units are degrees
        east and degrees north).
    steps: int, optional
        The number of points along the geodesic between the start and the end point
        (including the end points) to use in the cross section. Defaults to 100.
    interp_type: str, optional
        The interpolation method, either 'linear' or 'nearest' (see
        `xarray.DataArray.interp()` for details). Defaults to 'linear'.
    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        The interpolated cross section, with new index dimension along the cross-section.
    See Also
    --------
    interpolate_to_slice, geodesic
    """
    # https://github.com/Unidata/MetPy/blob/main/src/metpy/interpolate/slices.py
    import xarray as xr

    if isinstance(data, xr.Dataset):
        # Recursively apply to dataset
        return data.map(
            cross_section, True, (start, end), steps=steps, interp_type=interp_type
        )
    elif data.ndim == 0:
        # This has no dimensions, so it is likely a projection variable. In any case, there
        # are no data here to take the cross section with. Therefore, do nothing.
        return data

    # Get the geodesic
    points_cross = geodesic(start, end, steps, crs=crs)

    # Return the interpolated data
    return interpolate_to_slice(data, points_cross, interp_type=interp_type)


# alias
transect = cross_section


def interpolate_to_slice(data, points, x="lon", y="lat", interp_type="linear"):
    """Obtain an interpolated slice through data using xarray.
    Utilizing the interpolation functionality in `xarray`, this function takes a slice the
    given data (currently only regular grids are supported), which is given as an
    `xarray.DataArray` so that we can utilize its coordinate metadata.
    Parameters
    ----------
    data: `xarray.DataArray` or `xarray.Dataset`
        Three- (or higher) dimensional field(s) to interpolate. The DataArray (or each
        DataArray in the Dataset) must have been parsed by MetPy and include both an x and
        y coordinate dimension.
    points: (N, 2) array_like
        A list of x, y points in the data projection at which to interpolate the data
    interp_type: str, optional
        The interpolation method, either 'linear' or 'nearest' (see
        `xarray.DataArray.interp()` for details). Defaults to 'linear'.
    Returns
    -------
    `xarray.DataArray` or `xarray.Dataset`
        The interpolated slice of data, with new index dimension of size N.
    See Also
    --------
    cross_section
    """
    import xarray as xr

    data_sliced = data.interp(
        {
            x: xr.DataArray(points[:, 0], dims="index"),
            y: xr.DataArray(points[:, 1], dims="index"),
        },
        method=interp_type,
    )
    data_sliced.coords["index"] = range(len(points))

    return data_sliced


def geodesic(start, end, steps, crs="EPSG:4326"):
    """Construct a geodesic path between two points.
    This function acts as a wrapper for the geodesic construction available in `pyproj`.
    Parameters
    ----------
    start: (2, ) array_like
        A longitude-latitude designating the start point of the geodesic (units are
        degrees east and degrees north).
    end: (2, ) array_like
        A longitude-latitude pair designating the end point of the geodesic (units are degrees
        east and degrees north).
    steps: int, optional
        The number of points along the geodesic between the start and the end point
        (including the end points).
    crs: `pyproj.CRS`
        PyProj Coordinate Reference System to use for the output. Default EPSG:4326.
    Returns
    -------
    `numpy.ndarray`
        The list of x, y points in the given CRS of length `steps` along the geodesic.
    See Also
    --------
    cross_section
    """
    from pyproj import Proj
    from pyproj.crs import CRS

    if isinstance(crs, str):
        crs = CRS(crs)

    g = crs.get_geod()
    p = Proj(crs)

    # Geod.npts only gives points *in between* the start and end, and we want to include
    # the endpoints.
    geodesic = np.concatenate([
        np.array(start)[None],
        np.array(g.npts(*start, *end, steps - 2)),
        np.array(end)[None],
    ]).transpose()
    return np.stack(p(geodesic[0], geodesic[1], inverse=False, radians=False), axis=-1)
