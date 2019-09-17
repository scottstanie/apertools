#! /usr/bin/env python
"""Author: Scott Staniewicz
Input/Output functions for loading/saving SAR data in binary formats
Email: scott.stanie@utexas.edu
"""

from __future__ import division, print_function
import datetime
import glob
import math
import json
import os
import re
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import sardem

from apertools import utils
import apertools.parsers
from apertools.log import get_log
logger = get_log()

# 2to3 compat.
try:
    basestring
except NameError:
    basestring = str

FLOAT_32_LE = np.dtype('<f4')
COMPLEX_64_LE = np.dtype('<c8')

SENTINEL_EXTS = ['.geo', '.cc', '.int', '.amp', '.unw', '.unwflat']
UAVSAR_EXTS = [
    '.int',
    '.mlc',
    '.slc',
    '.amp',
    '.cor',
    '.grd',
    '.unw',
    '.int.grd',
    '.unw.grd',
    '.cor.grd',
    '.cc.grd',
    '.amp1.grd',
    '.amp2.grd',
    '.amp.grd',
]
IMAGE_EXTS = ['.png', '.tif', '.tiff', '.jpg']

# Notes: .grd, .mlc can be either real or complex for UAVSAR,
# .amp files are real only for UAVSAR, complex for sentinel processing
# However, we label them as real here since we can tell .amp files
# are from sentinel if there exists .rsc files in the same dir
COMPLEX_EXTS = [
    '.int',
    '.slc',
    '.geo',
    '.cc',
    '.unw',
    '.unwflat',
    '.mlc',
    '.int.grd',
    '.unw.grd',
]
REAL_EXTS = [
    '.amp',
    '.cor',
    '.mlc',
    '.grd',
    '.cor.grd',
    '.amp1.grd',
    '.amp2.grd',
]  # NOTE: .cor might only be real for UAVSAR
# Note about UAVSAR Multi-look products:
# Will end with `_ML5X5.grd`, e.g., for 5x5 downsampled

ELEVATION_EXTS = ['.dem', '.hgt']

# These file types are not simple complex matrices: see load_stacked_img for detail
# .unwflat are same as .unw, but with a linear ramp removed
STACKED_FILES = ['.cc', '.unw', '.unwflat', '.unw.grd', '.cc.grd']
# real or complex for these depends on the polarization
UAVSAR_POL_DEPENDENT = ['.grd', '.mlc']

DATE_FMT = "%Y%m%d"

# Constants for dataset keys saved in .h5 files
MASK_FILENAME = "masks.h5"
INT_FILENAME = "int_stack.h5"
UNW_FILENAME = "unw_stack.h5"
CC_FILENAME = "cc_stack.h5"

# dataset names for general 3D stacks
STACK_DSET = "stack"
STACK_MEAN_DSET = "mean_stack"
STACK_FLAT_DSET = "deramped_stack"
STACK_FLAT_SHIFTED_DSET = "deramped_shifted_stack"

# Mask file datasets
GEO_MASK_DSET = "geo"
GEO_MASK_SUM_DSET = "geo_sum"
IGRAM_MASK_DSET = "igram"
IGRAM_MASK_SUM_DSET = "igram_sum"

DEM_RSC_DSET = "dem_rsc"

GEOLIST_DSET = "geo_dates"
INTLIST_DSET = "int_dates"


def load_file(filename,
              downsample=None,
              looks=None,
              rsc_file=None,
              rsc_data=None,
              ann_info=None,
              verbose=False,
              **kwargs):
    """Examines file type for real/complex and runs appropriate load

    Args:
        filename (str): path to the file to open
        rsc_file (str): path to a dem.rsc file (if Sentinel)
        rsc_data (str): preloaded dem.rsc file as a dict
        ann_info (dict): data parsed from annotation file (UAVSAR)
        downsample (int): rate at which to downsample the file
            None is equivalent to 1, no downsampling.
        looks (tuple[int, int]): downsample by taking looks
            None is equivalent to (1, 1), no downsampling.
        verbose (bool): print extra logging info while loading files

    Returns:
        ndarray: a 2D array of the data from a file

    Raises:
        ValueError: if sentinel files loaded without a .rsc file in same path
            to give the file width
    """
    if downsample:
        if (downsample < 1 or not isinstance(downsample, int)):
            raise ValueError("downsample must be a positive integer")
        looks = (downsample, downsample)
    elif looks:
        if any((r < 1 or not isinstance(r, int)) for r in looks):
            raise ValueError("looks values must be a positive integers")
    else:
        looks = (1, 1)

    ext = utils.get_file_ext(filename)
    # Pass through numpy files to np.load
    if ext == '.npy':
        return utils.take_looks(np.load(filename), *looks)

    if ext == '.geojson':
        with open(filename) as f:
            return json.load(f)

    # Elevation and rsc files can be immediately loaded without extra data
    if ext in ELEVATION_EXTS:
        return utils.take_looks(sardem.loading.load_elevation(filename), *looks)
    elif ext == '.rsc':
        return sardem.loading.load_dem_rsc(filename, **kwargs)
    elif ext == '.h5':
        with h5py.File(filename, "r") as f:
            if len(f.keys()) > 1:
                try:
                    return f[kwargs["dset"]][:]
                except KeyError:
                    print("sario.load for h5 requres `dset` kwarg")
                    raise
            else:
                return f[list(f)[0]][:]

    # Sentinel files should have .rsc file: check for dem.rsc, or elevation.rsc
    if rsc_data is None and rsc_file:
        rsc_data = sardem.loading.load_dem_rsc(rsc_file)

    if ext in IMAGE_EXTS:
        return np.array(Image.open(filename).convert("L"))  # L for luminance == grayscale

    if ext in SENTINEL_EXTS:
        rsc_file = rsc_file if rsc_file else find_rsc_file(filename, verbose=verbose)
        if rsc_file:
            rsc_data = sardem.loading.load_dem_rsc(rsc_file)

    if ext == '.grd':
        ext = _get_full_grd_ext(filename)

    # UAVSAR files have an annotation file for metadata
    if not ann_info and not rsc_data and ext in UAVSAR_EXTS:
        try:
            u = apertools.parsers.Uavsar(filename, verbose=verbose)
            ann_info = u.parse_ann_file()
        except ValueError:
            try:
                u = apertools.parsers.UavsarInt(filename, verbose=verbose)
                ann_info = u.parse_ann_file()
            except ValueError:
                print("Failed loading ann_info")
                pass

    if not ann_info and not rsc_data:
        raise ValueError("Need .rsc file or .ann file to load")

    if ext in STACKED_FILES:
        stacked = load_stacked_img(filename, rsc_data=rsc_data, ann_info=ann_info, **kwargs)
        return stacked[..., ::downsample, ::downsample]
    # having rsc_data implies that this is not a UAVSAR file, so is complex
    elif rsc_data or is_complex(filename=filename, ext=ext):
        return utils.take_looks(
            load_complex(filename, ann_info=ann_info, rsc_data=rsc_data), *looks)
    else:
        return utils.take_looks(load_real(filename, ann_info=ann_info, rsc_data=rsc_data), *looks)


# Make a shorter alias for load_file
load = load_file


def _get_full_grd_ext(filename):
    if any(e in filename for e in ('.int', '.unw', '.cor', '.cc', '.amp1', '.amp2', '.amp')):
        ext = '.' + '.'.join(filename.split('.')[-2:])
        logger.info("Using %s for full grd extension" % ext)
        return ext
    else:
        return '.grd'


def find_files(directory, search_term):
    """Searches for files in `directory` using globbing on search_term

    Path to file is also included.
    Returns in names sorted order.

    Examples:
    >>> import shutil, tempfile
    >>> temp_dir = tempfile.mkdtemp()
    >>> open(os.path.join(temp_dir, "afakefile.txt"), "w").close()
    >>> print('afakefile.txt' in find_files(temp_dir, "*.txt")[0])
    True
    >>> shutil.rmtree(temp_dir)
    """
    return sorted(glob.glob(os.path.join(directory, search_term)))


def find_rsc_file(filename=None, directory=None, verbose=False):
    if filename:
        directory = os.path.split(os.path.abspath(filename))[0]
    # Should be just elevation.dem.rsc (for .geo folder) or dem.rsc (for igrams)
    possible_rscs = find_files(directory, '*.rsc')
    if verbose:
        logger.info("Searching %s for rsc files", directory)
        logger.info("Possible rsc files:")
        logger.info(possible_rscs)
    if len(possible_rscs) < 1:
        logger.info("No .rsc file found in %s", directory)
        return None
        # raise ValueError("{} needs a .rsc file with it for width info.".format(filename))
    elif len(possible_rscs) > 1:
        raise ValueError("{} has multiple .rsc files in its directory: {}".format(
            filename, possible_rscs))
    return utils.fullpath(possible_rscs[0])


def _get_file_rows_cols(ann_info=None, rsc_data=None):
    """Wrapper function to find file width for different SV types"""
    if (not rsc_data and not ann_info) or (rsc_data and ann_info):
        raise ValueError("needs either ann_info or rsc_data (but not both) to find number of cols")
    elif rsc_data:
        return rsc_data['file_length'], rsc_data['width']
    elif ann_info:
        return ann_info['rows'], ann_info['cols']


def _assert_valid_size(data, cols):
    """Make sure the width of the image is valid for the data size

    Note that only width is considered- The number of rows is ignored
    """
    error_str = "Invalid number of cols (%s) for file size %s." % (cols, len(data))
    # math.modf returns (fractional remainder, integer remainder)
    assert math.modf(float(len(data)) / cols)[0] == 0, error_str


def load_real(filename, ann_info=None, rsc_data=None, dtype=FLOAT_32_LE):
    """Reads in real 4-byte per pixel files""

    Valid filetypes: See sario.REAL_EXTS

    Args:
        filename (str): path to the file to open
        rsc_data (dict): output from load_dem_rsc, gives width of file
        ann_info (dict): data parsed from UAVSAR annotation file

    Returns:
        ndarray: float32 values for the real 2D matrix

    """
    data = np.fromfile(filename, dtype)
    rows, cols = _get_file_rows_cols(ann_info=ann_info, rsc_data=rsc_data)
    _assert_valid_size(data, cols)
    return data.reshape([-1, cols])


def load_complex(filename, ann_info=None, rsc_data=None, dtype=FLOAT_32_LE):
    """Combines real and imaginary values from a filename to make complex image

    Valid filetypes: See sario.COMPLEX_EXTS

    Args:
        filename (str): path to the file to open
        rsc_data (dict): output from load_dem_rsc, gives width of file
        ann_info (dict): data parsed from UAVSAR annotation file

    Returns:
        ndarray: imaginary numbers of the combined floats (dtype('complex64'))
    """
    data = np.fromfile(filename, dtype)
    rows, cols = _get_file_rows_cols(ann_info=ann_info, rsc_data=rsc_data)
    _assert_valid_size(data, cols)

    real_data, imag_data = parse_complex_data(data, cols)
    return combine_real_imag(real_data, imag_data)


def load_stacked_img(filename,
                     rsc_data=None,
                     ann_info=None,
                     return_amp=False,
                     dtype=FLOAT_32_LE,
                     **kwargs):
    """Helper function to load .unw and .cor files

    Format is two stacked matrices:
        [[first], [second]] where the first "cols" number of floats
        are the first matrix, next "cols" are second, etc.
    For .unw height files, the first is amplitude, second is phase (unwrapped)
    For .cc correlation files, first is amp, second is correlation (0 to 1)

    Args:
        filename (str): path to the file to open
        rsc_data (dict): output from load_dem_rsc, gives width of file
        return_amp (bool): flag to request the amplitude data to be returned

    Returns:
        ndarray: dtype=float32, the second matrix (height, correlation, ...) parsed
        if return_amp == True, returns two ndarrays stacked along axis=0

    Example illustrating how strips of data alternate:
    reading unw (unwrapped phase) data

    data = np.fromfile('20141128_20150503.unw', '<f4')

    # The first section of data is amplitude data
    # The amplitude has a different, larger range of values
    amp = data[:cols]
    print(np.max(amp), np.min(amp))
    # Output: (27140.396, 118.341095)

    # The next part of the data is a line of phases:
    phase = data[cols:2*cols])
    print(np.max(phase), np.min(phase))
    # Output: (8.011558, -2.6779003)
    """
    data = np.fromfile(filename, dtype)
    rows, cols = _get_file_rows_cols(rsc_data=rsc_data, ann_info=ann_info)
    _assert_valid_size(data, cols)

    first = data.reshape((rows, 2 * cols))[:, :cols]
    second = data.reshape((rows, 2 * cols))[:, cols:]
    if return_amp:
        return np.stack((first, second), axis=0)
    else:
        return second


def is_complex(filename=None, ext=None):
    """Helper to determine if file data is real or complex

    Uses https://uavsar.jpl.nasa.gov/science/documents/polsar-format.html for UAVSAR
    Note: differences between 3 polarizations for .mlc files: half real, half complex
    """
    if ext is None:
        ext = utils.get_file_ext(filename)

    if ext not in COMPLEX_EXTS and ext not in REAL_EXTS:
        raise ValueError('Invalid filetype for load_file: %s\n '
                         'Allowed types: %s' % (ext, ' '.join(COMPLEX_EXTS + REAL_EXTS)))

    if ext in UAVSAR_POL_DEPENDENT:
        # Check if filename has one of the complex polarizations
        return any(pol in filename for pol in apertools.parsers.Uavsar.COMPLEX_POLS)
    else:
        return ext in COMPLEX_EXTS


def parse_complex_data(complex_data, cols):
    """Splits a 1-D array of real/imag bytes to 2 square arrays"""
    # double check if I ever need rows
    real_data = complex_data[::2].reshape([-1, cols])
    imag_data = complex_data[1::2].reshape([-1, cols])
    return real_data, imag_data


def combine_real_imag(real_data, imag_data):
    """Combines two float data arrays into one complex64 array"""
    return real_data + 1j * imag_data


def save(filename, array, normalize=True, cmap="gray", preview=False, vmax=None, vmin=None):
    """Save the numpy array in one of known formats

    Args:
        filename (str): Output path to save file in
        array (ndarray): matrix to save
        normalize (bool): scale array to [-1, 1]
        cmap (str, matplotlib.cmap): colormap (if output is png/jpg and will be plotted)
        preview (bool): for png/jpg, display the image before saving
    Returns:
        None

    Raises:
        NotImplementedError: if file extension of filename not a known ext
    """

    def _is_little_endian():
        """All UAVSAR data products save in little endian byte order"""
        return sys.byteorder == 'little'

    def _force_float32(arr):
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(FLOAT_32_LE)
        elif np.issubdtype(arr.dtype, np.complexfloating):
            return arr.astype(COMPLEX_64_LE)
        else:
            return arr

    ext = utils.get_file_ext(filename)
    if ext == '.grd':
        ext = _get_full_grd_ext(filename)
    if ext == '.png':  # TODO: or ext == '.jpg':
        # Normalize to be between 0 and 1
        if normalize:
            array = array / np.max(np.abs(array))
            vmin, vmax = -1, 1
        logger.info("previewing with (vmin, vmax) = (%s, %s)" % (vmin, vmax))
        # from PIL import Image
        # im = Image.fromarray(array)
        # im.save(filename)
        if preview:
            plt.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.show(block=True)

        plt.imsave(filename, array, cmap=cmap, vmin=vmin, vmax=vmax, format=ext.strip('.'))

    elif (ext in COMPLEX_EXTS + REAL_EXTS + ELEVATION_EXTS) and (ext not in STACKED_FILES):
        # If machine order is big endian, need to byteswap (TODO: test on big-endian)
        # TODO: Do we need to do this at all??
        if not _is_little_endian():
            array.byteswap(inplace=True)

        _force_float32(array).tofile(filename)
    elif ext in STACKED_FILES:
        if array.ndim != 3:
            raise ValueError("Need 3D stack ([amp, data]) to save.")
        # first = data.reshape((rows, 2 * cols))[:, :cols]
        # second = data.reshape((rows, 2 * cols))[:, cols:]
        np.hstack((array[0], array[1])).astype(FLOAT_32_LE).tofile(filename)

    else:
        raise NotImplementedError("{} saving not implemented.".format(ext))


def save_hgt(filename, amp_data, height_data):
    save(filename, np.stack((amp_data, height_data), axis=0))


def load_stack(file_list=None, directory=None, file_ext=None, **kwargs):
    """Reads a set of images into a 3D ndarray

    Args:
        file_list (list[str]): list of file names to stack
        directory (str): alternative to file_name: path to a dir containing all files
            This will be loaded in ls-sorted order
        file_ext (str): If using `directory`, the ending type
            of files to read (e.g. '.unw')

    Returns:
        ndarray: 3D array of each file stacked
            1st dim is the index of the image: stack[0, :, :]
    """
    if file_list is None:
        if file_ext is None:
            raise ValueError("need file_ext if using `directory`")
        else:
            file_list = find_files(directory, "*" + file_ext)

    # Test load to get shape
    test = load(file_list[0], **kwargs)
    nrows, ncols = test.shape
    dtype = test.dtype
    out = np.empty((len(file_list), nrows, ncols), dtype=dtype)

    # Now lazily load the files and store in pre-allocated 3D array
    file_gen = (load(filename, **kwargs) for filename in file_list)
    for idx, img in enumerate(file_gen):
        out[idx] = img

    return out


def get_full_path(directory=None, filename=None, full_path=None):
    if full_path:
        directory, filename = os.path.split(os.path.abspath(full_path))
    else:
        full_path = os.path.join(directory, os.path.split(filename)[1])
    return directory, filename, full_path


def load_deformation(igram_path=".", filename='deformation.h5', full_path=None, n=None, dset=None):
    """Loads a stack of deformation images from igram_path

    if using the "deformation.npy" version, igram_path must also contain
    the "geolist.npy" file

    Args:
        igram_path (str): directory of .npy file
        filename (str): default='deformation.npy', a .npy file of a 3D ndarray
        n (int): only load the last `n` layers of the stack

    Returns:
        tuple[ndarray, ndarray]: geolist 1D array, deformation 3D array
    """
    igram_path, filename, full_path = get_full_path(igram_path, filename, full_path)

    if utils.get_file_ext(filename) == ".npy":
        return _load_deformation_npy(
            igram_path=igram_path, filename=filename, full_path=full_path, n=n)
    elif utils.get_file_ext(filename) in (".h5", "hdf5"):
        return _load_deformation_h5(
            igram_path=igram_path, filename=filename, full_path=full_path, n=n, dset=dset)
    else:
        raise ValueError("load_deformation only supported for .h5 or .npy")


def _load_deformation_h5(igram_path=None, filename=None, full_path=None, n=None, dset=None):
    igram_path, filename, full_path = get_full_path(igram_path, filename, full_path)
    try:
        with h5py.File(full_path, "r") as f:
            if dset is None:
                dset = list(f)[0]
            # TODO: get rid of these strings not as constants
            if n is not None:
                deformation = f[dset][-n:]
            else:
                deformation = f[dset][:]
            # geolist attr will be is a list of strings: need them as datetimes

    except (IOError, OSError) as e:
        logger.error("Can't load %s in path %s: %s", filename, igram_path, e)
        return None, None
    try:
        geolist = load_geolist_from_h5(full_path)
    except Exception as e:
        logger.error("Can't load geolist from %s in path %s: %s", filename, igram_path, e)
        geolist = None

    return geolist, deformation


def _load_deformation_npy(igram_path=None, filename=None, full_path=None, n=None):
    igram_path, filename, full_path = get_full_path(igram_path, filename, full_path)

    try:
        deformation = np.load(os.path.join(igram_path, filename))
        if n is not None:
            deformation = deformation[-n:]
        # geolist is a list of datetimes: encoding must be bytes
        geolist = np.load(
            os.path.join(igram_path, 'geolist.npy'), encoding='bytes', allow_pickle=True)
    except (IOError, OSError):
        logger.error("%s or geolist.npy not found in path %s", filename, igram_path)
        return None, None

    return geolist, deformation


def load_geolist_from_h5(h5file):
    with h5py.File(h5file, "r") as f:
        geolist_str = f[GEOLIST_DSET][()].astype(str)

    return parse_geolist_strings(geolist_str)


def load_intlist_from_h5(h5file):
    with h5py.File(h5file, "r") as f:
        date_pair_strs = f[INTLIST_DSET][:].astype(str)

    return parse_intlist_strings(date_pair_strs)


def parse_geolist_strings(geolist_str):
    return [_parse(g) for g in geolist_str]


def parse_intlist_strings(date_pairs):
    # If we passed filename YYYYmmdd_YYYYmmdd.int
    if isinstance(date_pairs, basestring):
        date_pairs = [date_pairs.strip('.int').split('_')[:2]]
    return [(_parse(early), _parse(late)) for early, late in date_pairs]


def _parse(datestr):
    return datetime.datetime.strptime(datestr, DATE_FMT).date()


def find_geos(directory=".", parse=True, filename=None):
    """Reads in the list of .geo files used, in time order

    Can also pass a filename containing .geo files as lines.

    Args:
        directory (str): path to the geolist file or directory
        parse (bool): output as parsed datetime tuples. False returns the filenames
        filename (string): name of a file with .geo filenames

    Returns:
        list[date]: the parse dates of each .geo used, in date order

    """
    if filename is not None:
        with open(filename) as f:
            geo_file_list = f.read().splitlines()
    else:
        geo_file_list = find_files(directory, "*.geo")

    if not parse:
        return geo_file_list

    # Stripped of path for parser
    geolist = [os.path.split(fname)[1] for fname in geo_file_list]
    if not geolist:
        return []
        # raise ValueError("No .geo files found in %s" % directory)

    if re.match(r'S1[AB]_\d{8}\.geo', geolist[0]):  # S1A_YYYYmmdd.geo
        return sorted([_parse(_strip_geoname(geo)) for geo in geolist])
    elif re.match(r'\d{8}', geolist[0]):  # YYYYmmdd , just a date string
        return sorted([_parse(geo) for geo in geolist])
    else:  # Full sentinel product name
        return sorted([apertools.parsers.Sentinel(geo).start_time.date() for geo in geolist])


def _strip_geoname(name):
    """Leaves just date from format S1A_YYYYmmdd.geo"""
    return name.replace('S1A_', '').replace('S1B_', '').replace('.geo', '')


def find_igrams(directory=".", parse=True, filename=None):
    """Reads the list of igrams to return dates of images as a tuple

    Args:
        directory (str): path to the igram directory
        parse (bool): output as parsed datetime tuples. False returns the filenames
        filename (string): name of a file with .geo filenames

    Returns:
        tuple(date, date) of (early, late) dates for all igrams (if parse=True)
            if parse=False: returns list[str], filenames of the igrams

    """
    if filename is not None:
        with open(filename) as f:
            igram_file_list = f.read().splitlines()
    else:
        igram_file_list = find_files(directory, "*.int")

    if parse:
        igram_fnames = [os.path.split(f)[1] for f in igram_file_list]
        date_pairs = [intname.strip('.int').split('_')[:2] for intname in igram_fnames]
        return parse_intlist_strings(date_pairs)
    else:
        return igram_file_list


def load_dem_from_h5(h5file=None, dset="dem_rsc"):
    with h5py.File(h5file, "r") as f:
        return json.loads(f[dset][()])


def save_dem_to_h5(h5file, dem_rsc, dset_name="dem_rsc", overwrite=True):
    if not check_dset(h5file, dset_name, overwrite):
        return

    with h5py.File(h5file, "a") as f:
        f[dset_name] = json.dumps(dem_rsc)


def save_geolist_to_h5(igram_path=None, out_file=None, overwrite=False, geo_date_list=None):
    if not check_dset(out_file, GEOLIST_DSET, overwrite):
        return

    if geo_date_list is None:
        geo_date_list, _ = load_geolist_intlist(igram_path, parse=True)

    logger.debug("Saving geo dates to %s / %s" % (out_file, GEOLIST_DSET))
    with h5py.File(out_file, "a") as f:
        # JSON gets messed from doing from julia to h5py for now
        # f[GEOLIST_DSET] = json.dumps(_geolist_to_str(geo_date_list))
        f[GEOLIST_DSET] = _geolist_to_str(geo_date_list)


def save_intlist_to_h5(igram_path=None, out_file=None, overwrite=False, int_date_list=None):
    if not check_dset(out_file, INTLIST_DSET, overwrite):
        return

    if int_date_list is None:
        _, int_date_list = load_geolist_intlist(igram_path)

    logger.info("Saving igram dates to %s / %s" % (out_file, INTLIST_DSET))
    with h5py.File(out_file, "a") as f:
        f[INTLIST_DSET] = _intlist_to_str(int_date_list)


def _geolist_to_str(geo_date_list):
    return np.array([d.strftime(DATE_FMT) for d in geo_date_list]).astype("S")


def _intlist_to_str(int_date_list):
    return np.array(
        [(a.strftime(DATE_FMT), b.strftime(DATE_FMT)) for a, b in int_date_list]).astype("S")


def load_geolist_intlist(directory, geolist_ignore_file=None, parse=True):
    """Load the geo_date_list and int_date_list from a directory with igrams

    Assumes that the .geo files are one diretory up from the igrams
    """
    int_date_list = find_igrams(directory, parse=parse)
    geo_date_list = find_geos(utils.get_parent_dir(directory), parse=parse)

    if geolist_ignore_file is not None:
        ignore_filepath = os.path.join(directory, geolist_ignore_file)
        geo_date_list, int_date_list = ignore_geo_dates(
            geo_date_list, int_date_list, ignore_file=ignore_filepath, parse=parse)
    return geo_date_list, int_date_list


def ignore_geo_dates(geo_date_list, int_date_list, ignore_file="geolist_missing.txt", parse=True):
    """Read extra file to ignore certain dates of interferograms"""
    ignore_geos = set(find_geos(ignore_file, parse=parse))
    logger.info("Ignoreing the following .geo dates:")
    logger.info(sorted(ignore_geos))
    valid_geos = [g for g in geo_date_list if g not in ignore_geos]
    valid_igrams = [i for i in int_date_list if i[0] not in ignore_geos and i[1] not in ignore_geos]
    return valid_geos, valid_igrams


def check_dset(h5file, dset_name, overwrite):
    """Returns false if the dataset exists and overwrite is False

    If overwrite is set to true, will delete the dataset to make
    sure a new one can be created
    """
    with h5py.File(h5file, "a") as f:
        if dset_name in f:
            logger.info("{dset} already exists in {file},".format(dset=dset_name, file=h5file))
            if overwrite:
                logger.info("Overwrite true: Deleting.")
                del f[dset_name]
            else:
                logger.info("Skipping.")
                return False

        return True


def load_mask(geo_date_list=None,
              perform_mask=True,
              deformation_filename=None,
              mask_filename="masks.h5",
              directory=None):
    # TODO: Dedupe this from the insar one
    if not perform_mask:
        return np.ma.nomask

    if directory is not None:
        _, _, mask_full_path = get_full_path(directory=directory, filename=mask_filename)
    else:
        mask_full_path = mask_filename
    if not os.path.exists(mask_full_path):
        logger.warning("{} doesnt exist, not masking".format(mask_full_path))
        return np.ma.nomask

    # If they pass a deformation .h5 stack, get only the dates actually used
    # instead of all possible dates stored in the mask stack
    if deformation_filename is not None:
        if directory is not None:
            deformation_filename = os.path.join(directory, deformation_filename)
            geo_date_list = load_geolist_from_h5(deformation_filename)

    # Get the indices of the mask layers that were used in the deformation stack
    all_geo_dates = load_geolist_from_h5(mask_full_path)
    if geo_date_list is None:
        used_bool_arr = np.full(len(all_geo_dates), True)
    else:
        used_bool_arr = np.array([g in geo_date_list for g in all_geo_dates])

    with h5py.File(mask_full_path) as f:
        # Maks a single mask image for any pixel that has a mask
        # Note: not using GEO_MASK_SUM_DSET since we may be sub selecting layers
        geo_dset = f[GEO_MASK_DSET]
        with geo_dset.astype(bool):
            stack_mask = np.sum(geo_dset[used_bool_arr, :, :], axis=0) > 0
        return stack_mask


def load_single_mask(int_date_string=None,
                     date_pair=None,
                     mask_filename=MASK_FILENAME,
                     int_date_list=None):
    """Load one mask from the `mask_filename`

    Can either pass a tuple of Datetimes in date_pair, or a string like
    `20170101_20170104.int` or `20170101_20170303` to int_date_string
    """
    if int_date_list is None:
        int_date_list = load_intlist_from_h5(mask_filename)

    if int_date_string is not None:
        # If the pass string with ., only take first part
        date_str_pair = int_date_string.split('.')[0].split('_')
        date_pair = parse_intlist_strings([date_str_pair])[0]

    with h5py.File(mask_filename, "r") as f:
        idx = int_date_list.index(date_pair)
        dset = f[IGRAM_MASK_DSET]
        with dset.astype(bool):
            return dset[idx]
