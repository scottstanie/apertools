#! /usr/bin/env python
"""Author: Scott Staniewicz
Input/Output functions for loading/saving SAR data in binary formats
Email: scott.stanie@utexas.edu
"""

from __future__ import division, print_function
import datetime
import fileinput
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
ALOS_EXTS = ['.slc', '.cc', '.int', '.amp', '.unw', '.unwflat']  # TODO: check these
BOOL_EXTS = ['.mask']
ROI_PAC_EXTS = ['.phs']
IMAGE_EXTS = ['.png', '.tif', '.tiff', '.jpg']

# phs?
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
    '.phs',  # GACOS, ROI_PAC?
    '.ztd',  # GACOS
]  # NOTE: .cor might only be real for UAVSAR
# Note about UAVSAR Multi-look products:
# Will end with `_ML5X5.grd`, e.g., for 5x5 downsampled

ELEVATION_EXTS = ['.dem', '.hgt']

# These file types are not simple complex matrices: see load_stacked_img for detail
# .unwflat are same as .unw, but with a linear ramp removed
STACKED_FILES = ['.cc', '.unw', '.unwflat', '.unw.grd', '.cc.grd']
# real or complex for these depends on the polarization
UAVSAR_POL_DEPENDENT = ['.grd', '.mlc']

BIL_FILES = STACKED_FILES  # Other name for storing binary interleaved by line
BIP_FILES = COMPLEX_EXTS

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
              rows=None,
              cols=None,
              verbose=False,
              **kwargs):
    """Examines file type for real/complex and runs appropriate load

    Args:
        filename (str): path to the file to open
        rsc_file (str): path to a dem.rsc file (if Sentinel)
        rsc_data (str): preloaded dem.rsc file as a dict
        ann_info (dict): data parsed from annotation file (UAVSAR)
        rows (int): manually pass number of rows (overrides rsc/ann data)
        cols (int): manually pass number of cols (overrides rsc/ann data)
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

    if ext in IMAGE_EXTS:
        return np.array(Image.open(filename).convert("L"))  # L for luminance == grayscale

    # Sentinel files should have .rsc file: check for dem.rsc, or elevation.rsc
    if rows is not None and cols is not None:
        rsc_data = {"rows": rows, "cols": cols, "width": cols, "height": rows}
    elif not rsc_file and os.path.exists(filename + ".rsc"):
        rsc_file = filename + ".rsc"
    elif not rsc_file and (ext in SENTINEL_EXTS or ext in ROI_PAC_EXTS or ext in BOOL_EXTS):
        # Try harder for .rsc
        rsc_file = find_rsc_file(filename, verbose=verbose)

    if rsc_file and rsc_data is None:
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

    if ext in BOOL_EXTS:
        return utils.take_looks(load_bool(filename, rsc_data=rsc_data, rows=rows, cols=cols),
                                *looks)
    elif ext in STACKED_FILES:
        stacked = load_stacked_img(filename,
                                   rsc_data=rsc_data,
                                   ann_info=ann_info,
                                   rows=rows,
                                   cols=cols,
                                   **kwargs)
        return stacked[..., ::downsample, ::downsample]
    elif is_complex(filename=filename, ext=ext):
        return utils.take_looks(
            load_complex(filename, ann_info=ann_info, rsc_data=rsc_data, rows=rows, cols=cols),
            *looks)
    else:
        return utils.take_looks(
            load_real(filename, ann_info=ann_info, rsc_data=rsc_data, rows=rows, cols=cols), *looks)


# Make a shorter alias for load_file
load = load_file


def _get_file_dtype(filename=None, ext=None):
    if ext is None:
        ext = utils.get_file_ext(filename)
    if ext in ELEVATION_EXTS:
        return np.int16
    elif ext in COMPLEX_EXTS:
        return np.complex64
    elif ext in REAL_EXTS:
        return np.float32
    else:
        raise NotImplementedError("Unknown file dtype for %s" % ext)


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
        errmsg = "multiple .rsc files directory: {}".format(possible_rscs[:5])
        if filename is None:
            raise ValueError(errmsg)

        fileonly = os.path.split(os.path.abspath(filename))[1]
        rscbases = [os.path.split(r)[1] for r in possible_rscs]
        rscdirs = [os.path.split(r)[0] for r in possible_rscs]
        if any(r.startswith(fileonly) for r in rscbases):  # Matching name
            possible_rscs = [
                os.path.join(b, r) for (b, r) in zip(rscdirs, rscbases) if r.startswith(fileonly)
            ]
        else:
            raise ValueError(errmsg)
    return utils.fullpath(possible_rscs[0])


def _get_file_rows_cols(rows=None, cols=None, ann_info=None, rsc_data=None):
    """Wrapper function to find file width for different satellite types"""
    if rows is not None and cols is not None:
        return rows, cols
    elif (not rsc_data and not ann_info) or (rsc_data and ann_info):
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


def load_real(filename, rows=None, cols=None, ann_info=None, rsc_data=None, dtype=FLOAT_32_LE):
    """Reads in real 4-byte per pixel files""

    Valid filetypes: See sario.REAL_EXTS

    Args:
        filename (str): path to the file to open
        rows (int): manually pass number of rows (overrides rsc/ann data)
        cols (int): manually pass number of cols (overrides rsc/ann data)
        rsc_data (dict): output from load_dem_rsc, gives width of file
        ann_info (dict): data parsed from UAVSAR annotation file

    Returns:
        ndarray: float32 values for the real 2D matrix

    """
    data = np.fromfile(filename, dtype)
    rows, cols = _get_file_rows_cols(rows=rows, cols=cols, ann_info=ann_info, rsc_data=rsc_data)
    _assert_valid_size(data, cols)
    return data.reshape([-1, cols])


def load_complex(filename, rows=None, cols=None, ann_info=None, rsc_data=None, dtype=FLOAT_32_LE):
    """Combines real and imaginary values from a filename to make complex image

    Valid filetypes: See sario.COMPLEX_EXTS

    Args:
        filename (str): path to the file to open
        rows (int): manually pass number of rows (overrides rsc/ann data)
        cols (int): manually pass number of cols (overrides rsc/ann data)
        rsc_data (dict): output from load_dem_rsc, gives width of file
        ann_info (dict): data parsed from UAVSAR annotation file

    Returns:
        ndarray: imaginary numbers of the combined floats (dtype('complex64'))
    """
    data = np.fromfile(filename, dtype)
    rows, cols = _get_file_rows_cols(rows=rows, cols=cols, ann_info=ann_info, rsc_data=rsc_data)
    _assert_valid_size(data, cols)

    real_data, imag_data = parse_complex_data(data, cols)
    return combine_real_imag(real_data, imag_data)


def load_bool(filename, rows=None, cols=None, ann_info=None, rsc_data=None, dtype=np.bool):
    """Load binary boolean image

    Args:
        filename (str): path to the file to open
        rows (int): manually pass number of rows (overrides rsc/ann data)
        cols (int): manually pass number of cols (overrides rsc/ann data)
        rsc_data (dict): output from load_dem_rsc, gives width of file
        ann_info (dict): data parsed from UAVSAR annotation file

    Returns:
        ndarray: imaginary numbers of the combined floats (dtype('complex64'))
    """
    data = np.fromfile(filename, dtype)
    rows, cols = _get_file_rows_cols(rows=rows, cols=cols, ann_info=ann_info, rsc_data=rsc_data)
    _assert_valid_size(data, cols)
    return data.reshape([-1, cols])


def load_stacked_img(filename,
                     rows=None,
                     cols=None,
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
        rows (int): manually pass number of rows (overrides rsc/ann data)
        cols (int): manually pass number of cols (overrides rsc/ann data)
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
    rows, cols = _get_file_rows_cols(rows=rows, cols=cols, ann_info=ann_info, rsc_data=rsc_data)
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


def save(filename, data, normalize=True, cmap="gray", preview=False, vmax=None, vmin=None):
    """Save the numpy array in one of known formats

    Args:
        filename (str): Output path to save file in
        data (ndarray): matrix to save
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
    if ext == ".rsc":
        with open(filename, "w") as f:
            f.write(sardem.loading.format_dem_rsc(data))
        return
    if ext == '.grd':
        ext = _get_full_grd_ext(filename)
    if ext == '.png':  # TODO: or ext == '.jpg':
        # Normalize to be between 0 and 1
        if normalize:
            data = data / np.max(np.abs(data))
            vmin, vmax = -1, 1
        logger.info("previewing with (vmin, vmax) = (%s, %s)" % (vmin, vmax))
        # from PIL import Image
        # im = Image.fromarray(data)
        # im.save(filename)
        if preview:
            plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.show(block=True)

        plt.imsave(filename, data, cmap=cmap, vmin=vmin, vmax=vmax, format=ext.strip('.'))

    elif ext in BOOL_EXTS:
        data.tofile(filename)
    elif (ext in COMPLEX_EXTS + REAL_EXTS + ELEVATION_EXTS) and (ext not in STACKED_FILES):
        # If machine order is big endian, need to byteswap (TODO: test on big-endian)
        # TODO: Do we need to do this at all??
        if not _is_little_endian():
            data.byteswap(inplace=True)

        _force_float32(data).tofile(filename)
    elif ext in STACKED_FILES:
        if data.ndim != 3:
            raise ValueError("Need 3D stack ([amp, data]) to save.")
        # first = data.reshape((rows, 2 * cols))[:, :cols]
        # second = data.reshape((rows, 2 * cols))[:, cols:]
        np.hstack((data[0], data[1])).astype(FLOAT_32_LE).tofile(filename)

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
        return _load_deformation_npy(igram_path=igram_path,
                                     filename=filename,
                                     full_path=full_path,
                                     n=n)
    elif utils.get_file_ext(filename) in (".h5", "hdf5"):
        return _load_deformation_h5(igram_path=igram_path,
                                    filename=filename,
                                    full_path=full_path,
                                    n=n,
                                    dset=dset)
    else:
        raise ValueError("load_deformation only supported for .h5 or .npy")


def _load_deformation_h5(igram_path=None, filename=None, full_path=None, n=None, dset=None):
    igram_path, filename, full_path = get_full_path(igram_path, filename, full_path)
    try:
        with h5py.File(full_path, "r") as f:
            if dset is None:
                dset = list(f)[0]
            if n is not None and n > 1:
                deformation = f[dset][-n:]
            else:
                deformation = f[dset][:]
            # geolist attr will be is a list of strings: need them as datetimes

    except (IOError, OSError) as e:
        logger.error("Can't load %s in path %s: %s", filename, igram_path, e)
        return None, None
    try:
        geolist = load_geolist_from_h5(full_path, dset=dset)
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
        geolist = np.load(os.path.join(igram_path, 'geolist.npy'),
                          encoding='bytes',
                          allow_pickle=True)
    except (IOError, OSError):
        logger.error("%s or geolist.npy not found in path %s", filename, igram_path)
        return None, None

    return geolist, deformation


def load_geolist_from_h5(h5file, dset=None):
    with h5py.File(h5file, "r") as f:
        if dset is None:
            geolist_str = f[GEOLIST_DSET][()].astype(str)
        else:
            geolist_str = f[dset].attrs[GEOLIST_DSET][()].astype(str)

    return parse_geolist_strings(geolist_str)


def load_intlist_from_h5(h5file, dset=None):
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
        return sorted([_parse(geo) for geo in geolist if geo])
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
    """Date pairs to Nx2 numpy array or strings"""
    return np.array([(a.strftime(DATE_FMT), b.strftime(DATE_FMT))
                     for a, b in int_date_list]).astype("S")


def intlist_to_filenames(int_date_list, ext=".int"):
    """Convert date pairs to list of string filenames"""
    return [
        "{}_{}{ext}".format(a.strftime(DATE_FMT), b.strftime(DATE_FMT), ext=ext)
        for a, b in int_date_list
    ]


def load_geolist_intlist(directory, geolist_ignore_file=None, parse=True):
    """Load the geo_date_list and int_date_list from a directory with igrams

    Assumes that the .geo files are one diretory up from the igrams
    """
    int_date_list = find_igrams(directory, parse=parse)
    geo_date_list = find_geos(utils.get_parent_dir(directory), parse=parse)

    if geolist_ignore_file is not None:
        ignore_filepath = os.path.join(directory, geolist_ignore_file)
        geo_date_list, int_date_list = ignore_geo_dates(geo_date_list,
                                                        int_date_list,
                                                        ignore_file=ignore_filepath,
                                                        parse=parse)
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
              dset=None,
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
            geo_date_list = load_geolist_from_h5(deformation_filename, dset=dset)

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


# ######### GDAL FUNCTIONS ##############


def save_as_geotiff(outfile=None, array=None, rsc_data=None, nodata=0.0):
    """Save an array to a GeoTIFF using gdal

    Ref: https://gdal.org/tutorials/raster_api_tut.html#using-create
    """
    import gdal
    rows, cols = array.shape
    if rsc_data is not None and (rows != rsc_data["file_length"] or cols != rsc_data["width"]):
        raise ValueError("rsc_data ({}, {}) does not match array shape: ({}, {})".format(
            (rsc_data["file_length"], rsc_data["width"], rows, cols)))

    driver = gdal.GetDriverByName('GTiff')

    gdal_dtype = numpy_to_gdal_type(array.dtype)
    out_raster = driver.Create(outfile, xsize=cols, ysize=rows, bands=1, eType=gdal_dtype)

    if rsc_data is not None:
        # Set geotransform (based on rsc data) and projection
        out_raster.SetGeoTransform(rsc_to_geotransform(rsc_data))
    srs = gdal.osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    out_raster.SetProjection(srs.ExportToWkt())

    band = out_raster.GetRasterBand(1)
    band.WriteArray(array)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    band = None
    out_raster = None


def save_as_vrt(filename=None,
                array=None,
                rows=None,
                cols=None,
                dtype=None,
                outfile=None,
                rsc_file=None,
                rsc_data=None,
                interleave=None,
                band=None,
                num_bands=None):
    """

    VRT options:
    SourceFilename: The name of the raw file containing the data for this band.
        The relativeToVRT attribute can be used to indicate if the
        SourceFilename is relative to the .vrt file (1) or not (0).
    ImageOffset: The offset in bytes to the beginning of the first pixel of
        data of this image band. Defaults to zero.
    PixelOffset: The offset in bytes from the beginning of one pixel and
        the next on the same line. In packed single band data this will be
        the size of the dataType in bytes.
    LineOffset: The offset in bytes from the beginning of one scanline of data
        and the next scanline of data. In packed single band data this will
        be PixelOffset * rasterXSize.

    Ref: https://gdal.org/drivers/raster/vrt.html#vrt-descriptions-for-raw-files
    """
    import gdal
    outfile = outfile or (filename + ".vrt")
    if outfile is None:
        raise ValueError("Need outfile or filename to save")

    # Set geotransform (based on rsc data) and projection
    if rsc_data is None:
        if rsc_file is None:
            # rsc_file = rsc_file if rsc_file else find_rsc_file(filename)
            print("Warning: No .rsc file or data given")
            geotrans = None
        else:
            rsc_data = load(rsc_file)

    if array is not None:
        dtype = array.dtype
        rows, cols = array.shape[-2:]

    if rsc_data is not None:
        rows, cols = _get_file_rows_cols(rsc_data=rsc_data)
        if array is not None:
            assert (rows, cols) == array.shape[-2:]

    if dtype is None:
        dtype = _get_file_dtype(filename)

    bytes_per_pix = np.dtype(dtype).itemsize
    total_bytes = os.path.getsize(filename)
    assert rows == int(total_bytes / bytes_per_pix / cols)
    # assert total_bytes == bytes_per_pix * rows * cols

    vrt_driver = gdal.GetDriverByName("VRT")

    # out_raster = vrt_driver.Create(outfile, xsize=cols, ysize=rows, bands=1, eType=gdal_dtype)
    out_raster = vrt_driver.Create(outfile, xsize=cols, ysize=rows, bands=0)

    if rsc_data is not None:
        geotrans = rsc_to_geotransform(rsc_data)
        out_raster.SetGeoTransform(geotrans)
    else:
        print("Warning: No GeoTransform could be made/set")

    srs = gdal.osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    out_raster.SetProjection(srs.ExportToWkt())

    if interleave is None or num_bands is None:
        interleave, num_bands = get_interleave(filename, num_bands=num_bands)
    if band is None:
        band = 2 if utils.get_file_ext(filename) in STACKED_FILES else 1

    image_offset, pixel_offset, line_offset = get_offsets(
        dtype,
        interleave,
        band,
        cols,
        rows,
        num_bands,
    )
    options = [
        'subClass=VRTRawRasterBand',
        # split, since relative to file, so remove directory name
        'SourceFilename={}'.format(os.path.split(filename)[1]),
        'relativeToVRT=1',  # location of file: make it relative to the VRT file
        'ImageOffset={}'.format(image_offset),
        'PixelOffset={}'.format(pixel_offset),
        'LineOffset={}'.format(line_offset),
        # 'ByteOrder=LSB'
    ]
    gdal_dtype = numpy_to_gdal_type(dtype)
    # print("gdal dtype", gdal_dtype, dtype)
    out_raster.AddBand(gdal_dtype, options)
    out_raster = None  # Force write

    # if geotrans is not None:
    # create_derived_band(outfile, rows, cols, geotrans, func="log10")
    # create_derived_band(outfile, rows, cols, geotrans, func="phase")
    return


def create_derived_band(src_filename, outfile=None, src_dtype="CFloat32", desc=None, func="log10"):
    import gdal
    # For new outfile, only have one .vrt extension
    if outfile is None:
        outfile = "{}.{}.vrt".format(src_filename.replace(".vrt", ""), func)
    desc = desc or "{func} of {filename}".format(func=func, filename=src_filename)
    srs = gdal.osr.SpatialReference()
    srs.SetWellKnownGeogCS("WGS84")
    srs_string = srs.ExportToWkt()

    f_src = gdal.Open(src_filename)
    geotrans = f_src.GetGeoTransform()
    rows = f_src.RasterYSize
    cols = f_src.RasterXSize
    f_src = None

    derived_vrt_template = """<VRTDataset rasterXSize="{cols}" rasterYSize="{rows}">
  <SRS> {srs} </SRS>
  <GeoTransform> {geotrans} </GeoTransform>
  <VRTRasterBand dataType="Float32" band="1" subClass="VRTDerivedRasterBand">
    <Description> {desc} </Description>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{src_filename}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
    <PixelFunctionType>{func}</PixelFunctionType>
    <SourceTransferType>{src_dtype}</SourceTransferType>
    <NoDataValue>-inf</NoDataValue>
  </VRTRasterBand>
</VRTDataset>
""".format(
        src_filename=src_filename,
        geotrans=",".join((str(n) for n in geotrans)),
        srs=srs_string,
        rows=rows,
        cols=cols,
        desc=desc,
        func=func,
        src_dtype=src_dtype,
    )
    # colors=make_cmy_colortable())
    with open(outfile, "w") as f:
        f.write(derived_vrt_template)

    # Colors:
    # ds = gdal.Open(outfile)
    # colors = make_cmy_colortable()
    # band = ds.GetRasterBand(1)
    # band.SetRasterColorTable(colors)
    # band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    return


def get_interleave(filename, num_bands=None):
    """Returns band interleave format, and number of bands"""
    if num_bands == 1:
        # 1 band is always same: its just all pixels in a row
        return "BIP", 1

    ext = utils.get_file_ext(filename)
    if ext in BIL_FILES:
        interleave, num_bands = "BIL", 2
    # TODO: the .amp files are actually BIP with 2 bands...
    elif ext in BIP_FILES:
        interleave, num_bands = "BIP", 1
    else:
        raise ValueError("Unknown band interleave format (BIP/BIL) for {}".format(filename))
    return interleave, num_bands


def get_offsets(dtype, interleave, band, width, length, num_bands):
    """
    From ISCE Image.py:
    """
    bytes_per_pix = np.dtype(dtype).itemsize
    if interleave == "BIL":
        return (
            band * width * bytes_per_pix,  # ImageOFfset
            bytes_per_pix,  # PixelOffset
            num_bands * width * bytes_per_pix,  # LineOffset
        )
    elif interleave == "BIP":
        return (
            band * bytes_per_pix,
            num_bands * bytes_per_pix,
            num_bands * width * bytes_per_pix,
        )
    elif interleave == "BSQ":
        return (
            band * width * length * bytes_per_pix,
            bytes_per_pix,
            width * bytes_per_pix,
        )
    else:
        raise ValueError("Unknown interleave: %s" % interleave)


def rsc_to_geotransform(rsc_data):

    # See here for geotransform info
    # https://gdal.org/user/raster_data_model.html#affine-geotransform
    # NOTE: gdal standard is to reference pixel by top left corner,
    # while the SAR .rsc stuff wants center of pixel
    # Xgeo = GT(0) + Xpixel*GT(1) + Yline*GT(2)
    # Ygeo = GT(3) + Xpixel*GT(4) + Yline*GT(5)

    # So for us, this means we have
    # X0 = trans[0] + .5*trans[1] + (.5*trans[2])
    # Y0 = trans[3] + (.5*trans[4]) + .5*trans[5]
    # where trans[2], trans[4] are 0s for north-up rasters

    x_step = rsc_data["x_step"]
    y_step = rsc_data["y_step"]
    X0 = rsc_data["x_first"] - 0.5 * x_step
    Y0 = rsc_data["y_first"] - 0.5 * y_step
    return (X0, x_step, 0.0, Y0, 0.0, y_step)


def set_unit(filename, unit="cm"):
    from osgeo import gdalconst
    import gdal
    go = gdal.Open(filename, gdalconst.GA_Update)
    b1 = go.GetRasterBand(1)
    b1.SetUnitType(unit)
    b1 = None
    go = None


def cmy_colors():
    # Default cyclic colormap from isce/mdx, provided by Piyush Agram, Jan 2020
    # generate the color list
    rgbs = np.zeros((256, 3), dtype=np.uint8)

    for kk in range(85):
        rgbs[kk, 0] = kk * 3
        rgbs[kk, 1] = 255 - kk * 3
        rgbs[kk, 2] = 255

    rgbs[85:170, 0] = rgbs[0:85, 2]
    rgbs[85:170, 1] = rgbs[0:85, 0]
    rgbs[85:170, 2] = rgbs[0:85, 1]

    rgbs[170:255, 0] = rgbs[0:85, 1]
    rgbs[170:255, 1] = rgbs[0:85, 2]
    rgbs[170:255, 2] = rgbs[0:85, 0]

    rgbs[255, 0] = 0
    rgbs[255, 1] = 255
    rgbs[255, 2] = 255

    rgbs = np.roll(rgbs, int(256 / 2 - 214), axis=0)  # shift green to the center
    rgbs = np.flipud(rgbs)  # flip up-down so that orange is in the later half (positive)
    return rgbs


def make_cmy_colortable():
    import gdal
    # create color table
    colors = gdal.ColorTable()

    rgbs = cmy_colors()
    rgbs = [rgbs[0], rgbs[42], rgbs[84], rgbs[126], rgbs[168], rgbs[200], rgbs[255]]
    vals = np.linspace(-np.pi, np.pi, len(rgbs))
    # set color for each value
    # out = "<ColorTable>\n"
    for (val, (r, g, b)) in zip(vals, rgbs):
        print(int(val))
        colors.SetColorEntry(int(val), (r, g, b))
        # out += '<Entry c1="{}" c2="{}" c3="{}" c4="255"/>\n'.format(r, g, b)

    # out += "</ColorTable>\n"
    return colors
    # return out


#     def memMap(self, mode='r', band=None):
#         if self.scheme.lower() == 'bil':
#             immap = np.memmap(self.filename, self.toNumpyDataType(), mode,
#                             shape=(self.coord2.coordSize , self.bands, self.coord1.coordSize))
#             if band is not None:
#                 immap = immap[:, band, :]
#         elif self.scheme.lower() == 'bip':
#             immap = np.memmap(self.filename, self.toNumpyDataType(), mode,
#                               shape=(self.coord2.coordSize, self.coord1.coordSize, self.bands))
#             if band is not None:
#                 immap = immap[:, :, band]
#         elif self.scheme.lower() == 'bsq':
#             immap = np.memmap(self.filename, self.toNumpyDataType(), mode,
#                         shape=(self.bands, self.coord2.coordSize, self.coord1.coordSize))
#             if band is not None:
#                 immap = immap[band, :, :]
#         return immap


def numpy_to_gdal_type(np_dtype):
    from osgeo import gdal_array, gdalconst
    if np.issubdtype(bool, np_dtype):
        return gdalconst.GDT_Byte
    # Wrap in np.dtype in case string is passed
    return gdal_array.NumericTypeCodeToGDALTypeCode(np.dtype(np_dtype))


def gdal_to_numpy_type(gdal_dtype=None, band=None):
    from osgeo import gdal_array
    if gdal_dtype is None:
        gdal_dtype = band.DataType
    return gdal_array.GDALTypeCodeToNumericTypeCode(gdal_dtype)


# TODO: this dont work to add a colorbar to grayscale tif...
def testt(fn):
    import gdal
    ds = gdal.Open(fn, 1)
    band = ds.GetRasterBand(1)

    # create color table
    colors = gdal.ColorTable()

    # set color for each value
    colors.SetColorEntry(1, (112, 153, 89))
    colors.SetColorEntry(2, (242, 238, 162))
    colors.SetColorEntry(3, (242, 206, 133))
    colors.SetColorEntry(4, (194, 140, 124))
    colors.SetColorEntry(5, (214, 193, 156))

    # set color table and color interpretation
    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)


def make_unw_vrt(unw_filelist=None, directory=None, output="unw_stack.vrt", ext=".unw"):
    import gdal
    if unw_filelist is None:
        unw_filelist = glob.glob(os.path.join(directory, "*" + ext))

    gdal.BuildVRT(output, unw_filelist, separate=True, srcNodata="nan 0.0")
    # But we want the 2nd band (not an option on build for some reason)
    with fileinput.FileInput(output, inplace=True) as f:
        for line in f:
            print(line.replace("<SourceBand>1</SourceBand>", "<SourceBand>2</SourceBand>"), end='')
