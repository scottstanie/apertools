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
import pprint
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image
import sardem

from apertools import parsers, utils
from apertools.log import get_log
logger = get_log()

FLOAT_32_LE = np.dtype('<f4')
COMPLEX_64_LE = np.dtype('<c8')

SENTINEL_EXTS = ['.geo', '.cc', '.int', '.amp', '.unw', '.unwflat']
UAVSAR_EXTS = ['.int', '.mlc', '.slc', '.amp', '.cor', '.grd', '.unw']
IMAGE_EXTS = ['.png', '.tif', '.tiff', '.jpg']

# Notes: .grd, .mlc can be either real or complex for UAVSAR,
# .amp files are real only for UAVSAR, complex for sentinel processing
# However, we label them as real here since we can tell .amp files
# are from sentinel if there exists .rsc files in the same dir
COMPLEX_EXTS = ['.int', '.slc', '.geo', '.cc', '.unw', '.unwflat', '.mlc', '.grd']
REAL_EXTS = ['.amp', '.cor', '.mlc', '.grd']  # NOTE: .cor might only be real for UAVSAR
# Note about UAVSAR Multi-look products:
# Will end with `_ML5X5.grd`, e.g., for 5x5 downsampled

ELEVATION_EXTS = ['.dem', '.hgt']

# These file types are not simple complex matrices: see load_stacked_img for detail
# .unwflat are same as .unw, but with a linear ramp removed
STACKED_FILES = ['.cc', '.unw', '.unwflat', '.unw.grd', '.cor.grd', '.cc.grd']
# real or complex for these depends on the polarization
UAVSAR_POL_DEPENDENT = ['.grd', '.mlc']


def load_file(filename,
              downsample=None,
              looks=None,
              rsc_file=None,
              ann_info=None,
              verbose=False,
              **kwargs):
    """Examines file type for real/complex and runs appropriate load

    Args:
        filename (str): path to the file to open
        rsc_file (str): path to a dem.rsc file (if Sentinel)
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

    # Sentinel files should have .rsc file: check for dem.rsc, or elevation.rsc
    rsc_data = None
    if rsc_file:
        rsc_data = sardem.loading.load_dem_rsc(rsc_file)

    if ext in IMAGE_EXTS:
        return np.array(Image.open(filename).convert("L"))  # L for luminance == grayscale

    if ext in SENTINEL_EXTS:
        rsc_file = rsc_file if rsc_file else find_rsc_file(filename, verbose=verbose)
        if rsc_file:
            rsc_data = sardem.loading.load_dem_rsc(rsc_file)

    # UAVSAR files have an annotation file for metadata
    if not ann_info and not rsc_data and ext in UAVSAR_EXTS:
        try:
            u = parsers.Uavsar(filename, verbose=verbose)
            ann_info = u.parse_ann_file()
        except ValueError:
            try:
                u = parsers.UavsarInt(filename, verbose=verbose)
                ann_info = u.parse_ann_file()
            except ValueError:
                print("Failed loading ann_info")
                pass

    if not ann_info and not rsc_file:
        raise ValueError("Need .rsc file or .ann file to load")

    if ext in STACKED_FILES:
        stacked = load_stacked_img(filename, rsc_data, **kwargs)
        return stacked[..., ::downsample, ::downsample]
    # having rsc_data implies that this is not a UAVSAR file, so is complex
    elif rsc_data or is_complex(filename):
        return utils.take_looks(load_complex(filename, ann_info=ann_info, rsc_data=rsc_data),
                                *looks)
    else:
        return utils.take_looks(load_real(filename, ann_info=ann_info, rsc_data=rsc_data), *looks)


# Make a shorter alias for load_file
load = load_file


def find_files(directory, search_term):
    """Searches for files in `directory` using globbing on search_term

    Path to file is also included.

    Examples:
    >>> import shutil, tempfile
    >>> temp_dir = tempfile.mkdtemp()
    >>> open(os.path.join(temp_dir, "afakefile.txt"), "w").close()
    >>> print('afakefile.txt' in find_files(temp_dir, "*.txt")[0])
    True
    >>> shutil.rmtree(temp_dir)
    """
    return glob.glob(os.path.join(directory, search_term))


def find_rsc_file(filename=None, basepath=None, verbose=False):
    if filename:
        basepath = os.path.split(os.path.abspath(filename))[0]
    # Should be just elevation.dem.rsc (for .geo folder) or dem.rsc (for igrams)
    possible_rscs = find_files(basepath, '*.rsc')
    if verbose:
        logger.info("Searching %s for rsc files", basepath)
        logger.info("Possible rsc files:")
        logger.info(possible_rscs)
    if len(possible_rscs) < 1:
        logger.info("No .rsc file found in %s", basepath)
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


def load_real(filename, ann_info=None, rsc_data=None):
    """Reads in real 4-byte per pixel files""

    Valid filetypes: See sario.REAL_EXTS

    Args:
        filename (str): path to the file to open
        rsc_data (dict): output from load_dem_rsc, gives width of file
        ann_info (dict): data parsed from UAVSAR annotation file

    Returns:
        ndarray: float32 values for the real 2D matrix

    """
    data = np.fromfile(filename, FLOAT_32_LE)
    rows, cols = _get_file_rows_cols(ann_info=ann_info, rsc_data=rsc_data)
    _assert_valid_size(data, cols)
    return data.reshape([-1, cols])


def load_complex(filename, ann_info=None, rsc_data=None):
    """Combines real and imaginary values from a filename to make complex image

    Valid filetypes: See sario.COMPLEX_EXTS

    Args:
        filename (str): path to the file to open
        rsc_data (dict): output from load_dem_rsc, gives width of file
        ann_info (dict): data parsed from UAVSAR annotation file

    Returns:
        ndarray: imaginary numbers of the combined floats (dtype('complex64'))
    """
    data = np.fromfile(filename, FLOAT_32_LE)
    rows, cols = _get_file_rows_cols(ann_info=ann_info, rsc_data=rsc_data)
    _assert_valid_size(data, cols)

    real_data, imag_data = parse_complex_data(data, cols)
    return combine_real_imag(real_data, imag_data)


def load_stacked_img(filename, rsc_data=None, ann_info=None, return_amp=False, **kwargs):
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
    data = np.fromfile(filename, FLOAT_32_LE)
    rows, cols = _get_file_rows_cols(rsc_data=rsc_data, ann_info=ann_info)
    _assert_valid_size(data, cols)

    first = data.reshape((rows, 2 * cols))[:, :cols]
    second = data.reshape((rows, 2 * cols))[:, cols:]
    if return_amp:
        return np.stack((first, second), axis=0)
    else:
        return second


def is_complex(filename):
    """Helper to determine if file data is real or complex

    Uses https://uavsar.jpl.nasa.gov/science/documents/polsar-format.html for UAVSAR
    Note: differences between 3 polarizations for .mlc files: half real, half complex
    """
    ext = utils.get_file_ext(filename)
    if ext not in COMPLEX_EXTS and ext not in REAL_EXTS:
        raise ValueError('Invalid filetype for load_file: %s\n '
                         'Allowed types: %s' % (ext, ' '.join(COMPLEX_EXTS + REAL_EXTS)))

    if ext in UAVSAR_POL_DEPENDENT:
        # Check if filename has one of the complex polarizations
        return any(pol in filename for pol in parsers.Uavsar.COMPLEX_POLS)
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


def save(filename, array, normalize=True, cmap="gray", preview=False):
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
    if ext == '.png':  # TODO: or ext == '.jpg':
        # Normalize to be between 0 and 1
        if normalize:
            array = array / np.max(np.abs(array))
            vmin, vmax = -1, 1
        else:
            vmin, vmax = None, None
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
            file_list = sorted(find_files(directory, "*" + file_ext))

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


def load_deformation(igram_path, filename='deformation.npy'):
    """Loads a stack of deformation images from igram_path

    igram_path must also contain the "geolist.npy" file

    Args:
        igram_path (str): directory of .npy file
        filename (str): default='deformation.npy', a .npy file of a 3D ndarray

    Returns:
        tuple[ndarray, ndarray]: geolist 1D array, deformation 3D array
    """
    try:
        deformation = np.load(os.path.join(igram_path, filename))
        # geolist is a list of datetimes: encoding must be bytes
        geolist = np.load(os.path.join(igram_path, 'geolist.npy'), encoding='bytes')
    except (IOError, OSError):
        logger.error("%s or geolist.npy not found in path %s", filename, igram_path)
        return None, None

    return geolist, deformation


def create_hdf5_stack(outfile_name=None,
                      compression=None,
                      file_list=None,
                      directory=None,
                      file_ext=None,
                      **kwargs):
    """Make stack as hdf5 file from a list of files

    Args:
        outfile_name (str): if none provided, creates a file `[file_ext]_stack.h5`

    Returns:
        outfile_name
    """
    if not outfile_name:
        if not file_ext:
            file_ext = utils.get_file_ext(file_list[0])
        outfile_name = "{fext}_stack.h5".format(fext=file_ext.strip("."))

    if utils.get_file_ext(outfile_name) not in (".h5", ".hdf5"):
        raise ValueError("outfile_name must end in .h5 or .hdf5")

    # TODO: do we want to replace the .unw files with .h5 files, then make a Virtual dataset?
    # layout = h5py.VirtualLayout(shape=(len(file_list), nrows, ncols), dtype=dtype)

    with h5py.File(outfile_name, "w") as hf:
        hf.create_dataset(
            "stack",
            data=load_stack(file_list=file_list, directory=directory, file_ext=file_ext, **kwargs),
        )
        # vsource = h5py.VirtualSource()

    return outfile_name


def _parse(datestr):
    return datetime.datetime.strptime(datestr, "%Y%m%d").date()


def _strip_geoname(name):
    """Leaves just date from format S1A_YYYYmmdd.geo"""
    return name.replace('S1A_', '').replace('S1B_', '').replace('.geo', '')


def read_geolist(filepath="./geolist", fnames_only=False):
    """Reads in the list of .geo files used, in time order

    Args:
        filepath (str): path to the geolist file or directory
        fnames_only (bool): default False. if true, return list of filenames

    Returns:
        list[date]: the parse dates of each .geo used, in date order

    """
    if os.path.isdir(filepath):
        filepath = os.path.join(filepath, 'geolist')

    with open(filepath) as f:
        if fnames_only:
            return [fname for fname in f.read().splitlines()]
        else:
            # Stripped of path for parser
            geolist = [os.path.split(geoname)[1] for geoname in f.read().splitlines()]

    if re.match(r'S1[AB]_\d{8}\.geo', geolist[0]):  # S1A_YYYYmmdd.geo
        return sorted([_parse(_strip_geoname(geo)) for geo in geolist])
    else:  # Full sentinel product name
        return sorted([parsers.Sentinel(geo).start_time.date() for geo in geolist])


def read_intlist(filepath="./intlist", parse=True):
    """Reads the list of igrams to return dates of images as a tuple

    Args:
        filepath (str): path to the intlist directory, or file
        parse (bool): output the intlist as parsed datetime tuples

    Returns:
        tuple(date, date) of master, slave dates for all igrams (if parse=True)
            if parse=False: returns list[str], filenames of the igrams

    """

    if os.path.isdir(filepath):
        filepath = os.path.join(filepath, 'intlist')

    with open(filepath) as f:
        intlist = f.read().splitlines()

    if parse:
        intlist = [intname.strip('.int').split('_') for intname in intlist]
        return [(_parse(master), _parse(slave)) for master, slave in intlist]
    else:
        dirname = os.path.dirname(filepath)
        return [os.path.join(dirname, igram) for igram in intlist]
