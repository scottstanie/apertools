#! /usr/bin/env python
"""Author: Scott Staniewicz
utils.py: Miscellaneous helper functions
Email: scott.stanie@utexas.edu
"""
from __future__ import division, print_function
import errno
import sys
import os
import subprocess
import numpy as np

import sardem
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


def mkdir_p(path):
    """Emulates bash `mkdir -p`, in python style
    Used for igrams directory creation
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def which(program):
    """Mimics UNIX which

    Used from https://stackoverflow.com/a/377028"""
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def take_looks(arr, row_looks, col_looks, separate_complex=False):
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
    if np.iscomplexobj(arr) and separate_complex:
        mag_looked = take_looks(np.abs(arr), row_looks, col_looks)
        phase_looked = take_looks(np.angle(arr), row_looks, col_looks)
        return mag_looked * np.exp(1j * phase_looked)

    new_rows, new_cols = _find_look_outsize(arr.shape, row_looks, col_looks)

    nrows, ncols = arr.shape
    row_cutoff = nrows % row_looks
    col_cutoff = ncols % col_looks

    if row_cutoff != 0:
        arr = arr[:-row_cutoff, :]
    if col_cutoff != 0:
        arr = arr[:, :-col_cutoff]
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype('float')

    return np.mean(arr.reshape(new_rows, row_looks, new_cols, col_looks), axis=(3, 1))


def _find_look_outsize(shape, row_looks, col_looks):
    nrows, ncols = shape
    new_rows, new_cols = shape[0] // row_looks, shape[1] // col_looks
    return new_rows, new_cols


def take_looks_gdal(outname, src_filename, row_looks, col_looks, format="ROI_PAC"):
    """Downsample an array on disk using gdal_translate

    Cuts off values if the size isn't divisible by num looks

    NOTE: For complex data, looks on the magnitude are done separately
    from looks on the phase

    See https://github.com/OSGeo/gdal/blob/master/gdal/swig/python/osgeo/gdal.py#L328
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
    import gdal
    from osgeo import gdalconst
    if row_looks == 1 and col_looks == 1:
        raise ValueError("Must take looks for file on disk")
    in_ds = gdal.Open(src_filename)
    shape = (in_ds.RasterYSize, in_ds.RasterXSize)  # (rows, cols)
    new_rows, new_cols = _find_look_outsize(shape, row_looks, col_looks)
    return gdal.Translate(outname,
                          in_ds,
                          height=new_rows,
                          width=new_cols,
                          format=format,
                          resampleAlg=gdalconst.GRIORA_Average)


def crossmul_gdal(outfile, file1, file2, row_looks, col_looks, format="ROI_PAC"):
    """Uses gdal_calc.py to multiply, then gdal_translate for looks"""
    tmp = "tmp.tif"
    cmd = """gdal_calc.py -A {f1} -B {f1} --outfile={tmp} \
            --calc="A * conj(B)" --NoDataValue=0 """.format(f1=file1, f2=file2, tmp=tmp)
    subprocess.check_call(cmd, shell=True)
    take_looks_gdal(outfile, tmp, row_looks, col_looks, format=format)
    os.remove(tmp)


def clip(image):
    """Convert float image to only range 0 to 1 (clips)"""
    return np.clip(np.abs(image), 0, 1)


def log(image):
    """Converts magnitude amplitude image to log scale"""
    if np.iscomplexobj(image):
        image = np.abs(image)
    return 20 * np.log10(image)


# Alias: convert
db = log


def mag(db_image):
    """Reverse of log/db: decibel to magnitude"""
    return 10**(db_image / 20)


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


def find_looks_taken(igram_path,
                     geo_path=None,
                     igram_dem_file="dem.rsc",
                     geo_dem_file="elevation.dem.rsc"):
    """Calculates how many looks from .geo files to .int files"""
    if geo_path is None:
        geo_path = os.path.dirname(os.path.abspath(igram_path))

    geo_dem_rsc = sardem.loading.load_dem_rsc(os.path.join(geo_path, geo_dem_file))

    igram_dem_rsc = sardem.loading.load_dem_rsc(os.path.join(igram_path, igram_dem_file))

    row_looks = geo_dem_rsc['file_length'] // igram_dem_rsc['file_length']
    col_looks = geo_dem_rsc['width'] // igram_dem_rsc['width']
    return row_looks, col_looks


def calc_upsample_rate(rsc_filename=None):
    """Find the rate of upsampling on an rsc file

    Args:
        rate (int): rate by which to upsample the DEM
        rsc_dict (str): Optional, the rsc data from Stitcher.create_dem_rsc()
        filepath (str): Optional, location of .dem.rsc file

    Note: Must supply only one of rsc_dict or rsc_filename

    Returns:
        tuple(float, float): (x spacing, y spacing)

    Raises:
        TypeError: if neither (or both) rsc_filename and rsc_dict are given

    """
    rsc_dict = sardem.loading.load_dem_rsc(filename=rsc_filename)
    default_spacing = 1.0 / 3600  # NASA SRTM uses 3600 pixels for 1 degree, or 30 m
    x_spacing = abs(rsc_dict['x_step'])
    y_spacing = abs(rsc_dict['y_step'])
    return default_spacing / x_spacing, default_spacing / y_spacing


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
    return (np.sum(arr == 0) / arr.size)


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
        raise TypeError('`shape` must be a sequence of integer')
    else:
        if shape.ndim > 1:
            raise ValueError('`shape` must be one-dimensional sequence of integer')
        if len(x.shape) != len(shape):
            raise ValueError("`shape` length doesn't match with input array dimensions")
        if np.any(shape <= 0):
            raise ValueError('`shape` cannot contain non-positive value')

    if step is None:
        step = np.ones(len(x.shape), np.intp)
    else:
        try:
            step = np.array(step, np.intp)
        except ValueError:
            raise TypeError('`step` must be a sequence of integer')
        else:
            if step.ndim > 1:
                raise ValueError('`step` must be one-dimensional sequence of integer')
            if len(x.shape) != len(step):
                raise ValueError("`step` length doesn't match with input array dimensions")
            if np.any(step <= 0):
                raise ValueError('`step` cannot contain non-positive value')

    o = (np.array(x.shape) - shape) // step + 1  # output shape
    if np.any(o <= 0):
        raise ValueError('window shape cannot larger than input array shape')

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
    window_size = window_size or 1
    if not isinstance(window_size, int) or window_size < 1:
        raise ValueError("Invalid window_size %s: must be odd positive int" % window_size)
    elif row > stack.shape[1] or col > stack.shape[2]:
        raise ValueError("(%s, %s) out of bounds reference for stack size %s" %
                         (row, col, stack.shape))

    if window_size % 2 == 0:
        window_size -= 1
        print("Making window_size an odd number (%s) to get square" % window_size)

    win_size = window_size // 2
    return func(stack[:,
                row - win_size:row + win_size + 1,
                col - win_size:col + win_size + 1], axis=(1, 2))  # yapf: disable


# Randoms using the sentinelapi
def find_slc_products(api, gj_obj, date_start, date_end, area_relation='contains'):
    """Query for Sentinel 1 SCL products with common options

    from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
    api = api = SentinelAPI(user, pw)
    pecosgeo = geojson_to_wkt(geojson.read_json('pecosBig.geojson'))
    find_slc_products(pecosgeo, '20150101', '20171230')

    Returns:
        OrderedDict: key = '528c0630-bbbf-4a95-8415-c55aa5ce915a', the sentinel
    """
    # area_relation : 'Intersection', 'Contains', 'IsWithin'
    # contains means that the Sentinel footprint completely contains your geojson object
    return api.query(gj_obj,
                     date=(date_start, date_end),
                     platformname='Sentinel-1',
                     producttype='SLC',
                     area_relation=area_relation)


def show_titles(products):
    return [p['title'] for p in products.values()]


def fullpath(path):
    """Expands ~ and returns an absolute path"""
    return os.path.abspath(os.path.expanduser(path))


def force_symlink(src, dest):
    """python equivalent to 'ln -f -s': force overwrite """
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


def get_cache_dir(force_posix=False, app_name="apertools"):
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
        path = os.path.join(os.path.expanduser('~/.' + app_name))
    if sys.platform == 'darwin':
        path = os.path.join(os.path.expanduser('~/Library/Application Support'), app_name)
    path = os.path.join(
        os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.cache')),
        app_name,
    )
    if not os.path.exists(path):
        os.makedirs(path)
    return path
