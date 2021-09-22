#! /usr/bin/env python
"""Author: Scott Staniewicz
Input/Output functions for loading/saving SAR data in binary formats
Email: scott.stanie@utexas.edu
"""

from __future__ import division, print_function
from collections.abc import Iterable
import datetime
import fileinput
from glob import glob
import math
import json
import os
import re
import sys
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import h5py

import apertools.utils
import apertools.parsers
from .demloading import format_dem_rsc, load_dem_rsc, load_elevation
from apertools.log import get_log

logger = get_log()

_take_looks = apertools.utils.take_looks

FLOAT_32_LE = np.dtype("<f4")
COMPLEX_64_LE = np.dtype("<c8")

SENTINEL_EXTS = [".geo", ".cc", ".int", ".amp", ".unw", ".unwflat"]
UAVSAR_EXTS = [
    ".int",
    ".mlc",
    ".slc",
    ".amp",
    ".cor",
    ".grd",
    ".unw",
    ".int.grd",
    ".unw.grd",
    ".cor.grd",
    ".cc.grd",
    ".amp1.grd",
    ".amp2.grd",
    ".amp.grd",
]
ALOS_EXTS = [".slc", ".cc", ".int", ".amp", ".unw", ".unwflat"]  # TODO: check these
BOOL_EXTS = [".mask", ".msk", ".wbd"]
UINT_EXTS = [".conncomp"]
ROI_PAC_EXTS = [".phs"]
GDAL_FORMATS = [".vrt", ".tif"]
IMAGE_EXTS = [".png", ".tif", ".tiff", ".jpg"]

# phs?
# Notes: .grd, .mlc can be either real or complex for UAVSAR,
# .amp files are real only for UAVSAR, complex for sentinel processing
# However, we label them as real here since we can tell .amp files
# are from sentinel if there exists .rsc files in the same dir
COMPLEX_EXTS = [
    ".int",
    ".slc",
    ".geo",
    ".unwflat",
    ".mlc",
    ".int.grd",
]
REAL_EXTS = [
    ".amp",
    ".cor",
    ".mlc",
    ".unw",
    ".grd",
    ".cc",
    ".cor.grd",
    ".amp1.grd",
    ".amp2.grd",
    ".unw.grd",
    ".phs",  # GACOS, ROI_PAC?
    ".ztd",  # GACOS
]  # NOTE: .cor might only be real for UAVSAR
# Note about UAVSAR Multi-look products:
# Will end with `_ML5X5.grd`, e.g., for 5x5 downsampled

ELEVATION_EXTS = [".dem", ".hgt"]

# These file types are not simple complex matrices: see load_stacked_img for detail
# .unwflat are same as .unw, but with a linear ramp removed
STACKED_FILES = [".cc", ".unw", ".unwflat"]
# real or complex for these depends on the polarization
UAVSAR_POL_DEPENDENT = [".grd", ".mlc"]

BIL_FILES = STACKED_FILES  # Other name for storing binary interleaved by line
BIP_FILES = COMPLEX_EXTS + ELEVATION_EXTS  # these are just single band ones

DATE_FMT = "%Y%m%d"

# Constants for dataset keys saved in .h5 files
MASK_FILENAME = "masks.h5"
IFG_FILENAME = "ifg_stack.h5"
UNW_FILENAME = "unw_stack.h5"
COR_FILENAME = "cor_stack.h5"

# dataset names for general 3D stacks
STACK_DSET = "stack"
STACK_MEAN_DSET = "stack_mean"
STACK_FLAT_SHIFTED_DSET = "stack_flat_shifted"

# Mask file datasets
SLC_MASK_DSET = "slc"
SLC_MASK_SUM_DSET = "slc_sum"
IFG_MASK_DSET = "ifg"
IFG_MASK_SUM_DSET = "ifg_sum"

SLCLIST_DSET = "slc_dates"
IFGLIST_DSET = "ifg_dates"

DEM_RSC_DSET = "dem_rsc"
STACK_FLAT_DSET = "stack_flat"

LOS_FILENAME = "los_enu.tif"
ISCE_GEOM_DIR = "geom_reference"
ISCE_SLC_DIR = "merged/SLC"
ISCE_MASK_FILES = ["waterMask.rdr", "shadowMask.rdr"]

# List of platforms where i've set up loading for their files
PLATFORMS = ("sentinel", "uavsar")


def load(
    filename,
    arr=None,
    downsample=None,
    looks=None,
    platform="sentinel",
    rsc_file=None,
    rsc_data=None,
    ann_info=None,
    rows=None,
    cols=None,
    use_gdal=False,
    verbose=False,
    **kwargs,
):
    """Examines file type for real/complex and runs appropriate load

    Args:
        filename (str): path to the file to open
        arr (ndarray): pre-allocated array of the correct size/dtype
            TODO: (error if wrong size/type?)
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
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file or directory: '{filename}'")

    if downsample:
        if downsample < 1 or not isinstance(downsample, int):
            raise ValueError("downsample must be a positive integer")
        looks = (downsample, downsample)
    elif looks:
        if any((r < 1 or not isinstance(r, int)) for r in looks):
            raise ValueError("looks values must be a positive integers")
    else:
        looks = (1, 1)

    ext = apertools.utils.get_file_ext(filename)
    # Pass through numpy files to np.load
    if ext == ".npy":
        return _take_looks(np.load(filename), *looks)

    if ext == ".geojson":
        with open(filename) as f:
            return json.load(f)

    # Pass though and load with gdal
    if ext in GDAL_FORMATS or use_gdal:
        # Use rasterio for easier loading of all bands into stack
        import rasterio as rio

        with rio.open(filename) as src:
            return src.read(kwargs.get("band"))

    # Elevation and rsc files can be immediately loaded without extra data
    if ext in ELEVATION_EXTS:
        return _take_looks(load_elevation(filename), *looks)
    elif ext == ".rsc":
        return _take_looks(load_dem_rsc(filename, **kwargs), *looks)
    elif ext == ".h5":
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
        try:
            from PIL import Image
        except ImportError:
            print("Need PIL installed to save as image. `pip install pillow`")
            raise
        return np.array(
            Image.open(filename).convert("L")
        )  # L for luminance == grayscale

    if ext == ".grd":
        ext = _get_full_grd_ext(filename)

    # Double check the platform if there's only one option
    if ext in UAVSAR_EXTS and ext not in SENTINEL_EXTS and not rsc_file:
        platform = "uavsar"

    # If it hasn't been loaded by now, it's probably a radar file type
    if platform == "sentinel":
        # Sentinel files should have .rsc file: check for dem.rsc, or elevation.rsc
        if rows is not None and cols is not None:
            rsc_data = {"rows": rows, "cols": cols, "width": cols, "height": rows}
        elif not rsc_file and os.path.exists(filename + ".rsc"):
            rsc_file = filename + ".rsc"
        elif not rsc_file and (
            ext in SENTINEL_EXTS or ext in ROI_PAC_EXTS or ext in BOOL_EXTS
        ):
            # Try harder for .rsc
            rsc_file = find_rsc_file(filename, verbose=verbose)
        if rsc_file and rsc_data is None:
            rsc_data = load_dem_rsc(rsc_file)
    elif platform == "uavsar":
        if rows is not None and cols is not None:
            ann_info = {"rows": rows, "cols": cols, "width": cols, "height": rows}
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
    else:
        raise NotImplementedError(f"platform choices are {PLATFORMS}")

    if not ann_info and not rsc_data:
        raise ValueError("Need .rsc file or .ann file to load")

    if ext in BOOL_EXTS:
        return _take_looks(
            load_binary_img(
                filename,
                arr=arr,
                rsc_data=rsc_data,
                rows=rows,
                cols=cols,
                dtype=np.bool,
            ),
            *looks,
        )
    if ext in UINT_EXTS:
        return _take_looks(
            load_binary_img(
                filename,
                arr=arr,
                rows=rows,
                cols=cols,
                rsc_data=rsc_data,
                dtype=np.uint8,
            ),
            *looks,
        )
    # Note on UAVSAR loading: they dont seem to do any stacked files
    elif ext in STACKED_FILES and platform == "sentinel":
        stacked = load_stacked_img(
            filename,
            arr=arr,
            rsc_data=rsc_data,
            ann_info=ann_info,
            rows=rows,
            cols=cols,
            looks=looks,
            **kwargs,
        )
        return stacked[..., ::downsample, ::downsample]
    elif is_complex(filename=filename, ext=ext):
        # Complex64
        return _take_looks(
            load_binary_img(
                filename,
                arr=arr,
                rows=rows,
                cols=cols,
                ann_info=ann_info,
                rsc_data=rsc_data,
                dtype=COMPLEX_64_LE,
            ),
            *looks,
        )
    else:
        # Real float32
        return _take_looks(
            load_binary_img(
                filename,
                arr=arr,
                rows=rows,
                cols=cols,
                ann_info=ann_info,
                rsc_data=rsc_data,
                dtype=FLOAT_32_LE,
            ),
            *looks,
        )


# Make a original alias of load/load_file
load_file = load


def _get_file_dtype(filename=None, ext=None):
    if ext is None:
        ext = apertools.utils.get_file_ext(filename)
    if ext in ELEVATION_EXTS:
        return np.int16
    elif ext in COMPLEX_EXTS:
        return np.complex64
    elif ext in REAL_EXTS:
        return np.float32
    else:
        raise NotImplementedError("Unknown file dtype for %s" % ext)


def _get_full_grd_ext(filename):
    if any(
        e in filename for e in (".int", ".unw", ".cor", ".cc", ".amp1", ".amp2", ".amp")
    ):
        ext = "." + ".".join(filename.split(".")[-2:])
        logger.info("Using %s for full grd extension" % ext)
        return ext
    else:
        return ".grd"


def find_rsc_file(filename=None, directory=None, verbose=False):
    if filename:
        directory = os.path.split(os.path.abspath(filename))[0]
    # Should be just elevation.dem.rsc (for slc folder) or dem.rsc (for igrams)
    possible_rscs = sorted(glob(os.path.join(directory, "*.rsc")))
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
                os.path.join(b, r)
                for (b, r) in zip(rscdirs, rscbases)
                if r.startswith(fileonly)
            ]
        else:
            raise ValueError(errmsg)
    return apertools.utils.fullpath(possible_rscs[0])


def _get_file_rows_cols(rows=None, cols=None, ann_info=None, rsc_data=None):
    """Wrapper function to find file width for different satellite types"""
    if rows is not None and cols is not None:
        return rows, cols
    elif (not rsc_data and not ann_info) or (rsc_data and ann_info):
        raise ValueError(
            "needs either ann_info or rsc_data (but not both) to find number of cols"
        )
    elif rsc_data:
        return rsc_data["file_length"], rsc_data["width"]
    elif ann_info:
        return ann_info["rows"], ann_info["cols"]


def _assert_valid_size(data, cols):
    """Make sure the width of the image is valid for the data size

    Note that only width is considered- The number of rows is ignored
    """
    error_str = "Invalid number of cols (%s) for file size %s." % (cols, len(data))
    # math.modf returns (fractional remainder, integer remainder)
    assert math.modf(float(data.size) / cols)[0] == 0, error_str


def _load_binary1d(
    filename, arr=None, dtype=None, rows=None, cols=None, ann_info=None, rsc_data=None
):
    if arr is not None:
        if dtype is not None:
            assert arr.dtype == np.dtype(dtype), f"{arr.dtype} != {dtype}"
        else:
            dtype = arr.dtype
        # https://github.com/numpy/numpy/blob/master/numpy/core/records.py#L896-L897
        with open(filename, "rb") as fd:
            fd.readinto(arr.data)
            data = arr
    else:
        data = np.fromfile(filename, dtype)

    rows, cols = _get_file_rows_cols(
        rows=rows, cols=cols, ann_info=ann_info, rsc_data=rsc_data
    )
    _assert_valid_size(data, cols)
    return data, rows, cols


def load_binary_img(
    filename, arr=None, rows=None, cols=None, ann_info=None, rsc_data=None, dtype=None
):
    """Loads a binary image into a numpy array

    Size given preference to `rows` and `cols`, or will parse from rsc_data/ann_info

    Args:
        filename (str): path to the file to open
        arr (ndarray): pre-allocated array of the correct size/dtype
        rows (int): manually pass number of rows (overrides rsc/ann data)
        cols (int): manually pass number of cols (overrides rsc/ann data)
        rsc_data (dict): output from load_dem_rsc, gives width of file
        ann_info (dict): data parsed from UAVSAR annotation file
        dtype (str, np.dtype): data type for array

    Returns:
        ndarray: values for the real 2D matrix with datatype `dtype`

    """
    data, rows, cols = _load_binary1d(
        filename,
        dtype=dtype,
        arr=arr,
        rows=rows,
        cols=cols,
        ann_info=ann_info,
        rsc_data=rsc_data,
    )
    return data.reshape([-1, cols])


def load_stacked_img(
    filename,
    arr=None,
    rows=None,
    cols=None,
    rsc_data=None,
    ann_info=None,
    return_amp=False,
    looks=(1, 1),
    dtype=FLOAT_32_LE,
    **kwargs,
):
    """Helper function to load .unw and .cor files from snaphu output

    Format is two stacked matrices:
        [[first], [second]] where the first "cols" number of floats
        are the first matrix, next "cols" are second, etc.
    Also called BIL, Band Interleaved by Line
    See http://webhelp.esri.com/arcgisdesktop/9.3/index.cfm?topicname=BIL,_BIP,_and_BSQ_raster_files
    for explantion

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
    """
    data, rows, cols = _load_binary1d(
        filename,
        dtype=dtype,
        arr=arr,
        rows=rows,
        cols=cols,
        ann_info=ann_info,
        rsc_data=rsc_data,
    )

    first = data.reshape((rows, 2 * cols))[:, :cols]
    second = data.reshape((rows, 2 * cols))[:, cols:]
    if return_amp:
        return np.stack(
            (_take_looks(first, *looks), _take_looks(second, *looks)), axis=0
        )
    else:
        return _take_looks(second, *looks)


def is_complex(filename=None, ext=None):
    """Helper to determine if file data is real or complex

    Uses https://uavsar.jpl.nasa.gov/science/documents/polsar-format.html for UAVSAR
    Note: differences between 3 polarizations for .mlc files: half real, half complex
    """
    if ext is None:
        ext = apertools.utils.get_file_ext(filename)

    if ext not in COMPLEX_EXTS and ext not in REAL_EXTS:
        raise ValueError(
            "Invalid filetype for load_file: %s\n "
            "Allowed types: %s" % (ext, " ".join(COMPLEX_EXTS + REAL_EXTS))
        )

    if ext in UAVSAR_POL_DEPENDENT:
        # Check if filename has one of the complex polarizations
        return any(pol in filename for pol in apertools.parsers.Uavsar.COMPLEX_POLS)
    else:
        return ext in COMPLEX_EXTS


def save(
    filename, data, normalize=True, cmap="gray", preview=False, vmax=None, vmin=None
):
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
    import matplotlib.pyplot as plt

    def _is_little_endian():
        """All UAVSAR data products save in little endian byte order"""
        return sys.byteorder == "little"

    def _force_float32(arr):
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(FLOAT_32_LE)
        elif np.issubdtype(arr.dtype, np.complexfloating):
            return arr.astype(COMPLEX_64_LE)
        else:
            return arr

    ext = apertools.utils.get_file_ext(filename)
    if ext == ".rsc":
        with open(filename, "w") as f:
            f.write(format_dem_rsc(data))
        return
    if ext == ".grd":
        ext = _get_full_grd_ext(filename)
    if ext == ".png":  # TODO: or ext == '.jpg':
        # Normalize to be between 0 and 1
        if normalize:
            data = data / np.max(np.abs(data))
            vmin, vmax = -1, 1
        logger.info("previewing with (vmin, vmax) = (%s, %s)" % (vmin, vmax))
        if preview:
            plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.show(block=True)

        plt.imsave(
            filename, data, cmap=cmap, vmin=vmin, vmax=vmax, format=ext.strip(".")
        )

    elif ext in BOOL_EXTS:
        data.tofile(filename)
    elif (ext in COMPLEX_EXTS + REAL_EXTS + ELEVATION_EXTS) and (
        ext not in STACKED_FILES
    ):
        # If machine order is big endian, need to byteswap (TODO: test on big-endian)
        # TODO: Do we need to do this at all??
        if not _is_little_endian():
            data.byteswap(inplace=True)

        _force_float32(data).tofile(filename)
    elif ext in STACKED_FILES:
        save_bil(filename, data, dtype=FLOAT_32_LE)

    else:
        raise NotImplementedError("{} saving not implemented.".format(ext))


def save_hgt(filename, amp_data, height_data):
    save(filename, np.stack((amp_data, height_data), axis=0))


def save_bil(filename, data, dtype=np.float32):
    """Save a Band interleaves (BIL) array

    also known as "ALT_LINE_DATA" in SNAPHU's documentation.
    If data is a 2D array (only 1 band given), will save the
    first band with all zeros of same size
    """
    if data.ndim != 3:
        # raise ValueError("Need 3D stack ([amp, data]) to save.")
        band1 = np.zeros(data.shape, dtype=dtype)
        band2 = data
    else:
        band1, band2 = data
    # first = data.reshape((rows, 2 * cols))[:, :cols]
    # second = data.reshape((rows, 2 * cols))[:, cols:]
    np.hstack((band1, band2)).astype(dtype).tofile(filename)


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
            file_list = sorted(glob(os.path.join(directory, file_ext)))

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


def load_deformation(
    igram_path=".", filename="deformation.h5", full_path=None, n=None, dset=None
):
    """Loads a stack of deformation images from igram_path

    if using the "deformation.npy" version, igram_path must also contain
    the "slclist.npy" file

    Args:
        igram_path (str): directory of .npy file
        filename (str): default='deformation.npy', a .npy file of a 3D ndarray
        n (int): only load the last `n` layers of the stack

    Returns:
        tuple[ndarray, ndarray]: slclist 1D array, deformation 3D array
    """
    igram_path, filename, full_path = get_full_path(igram_path, filename, full_path)

    if apertools.utils.get_file_ext(filename) == ".npy":
        return _load_deformation_npy(
            igram_path=igram_path, filename=filename, full_path=full_path, n=n
        )
    elif apertools.utils.get_file_ext(filename) in (".h5", "hdf5"):
        return _load_deformation_h5(
            igram_path=igram_path,
            filename=filename,
            full_path=full_path,
            n=n,
            dset=dset,
        )
    else:
        raise ValueError("load_deformation only supported for .h5 or .npy")


def _load_deformation_h5(
    igram_path=None, filename=None, full_path=None, n=None, dset=None
):
    igram_path, filename, full_path = get_full_path(igram_path, filename, full_path)
    try:
        with h5py.File(full_path, "r") as f:
            if dset is None:
                dset = list(f)[0]
            if n is not None and n > 1:
                deformation = f[dset][-n:]
            else:
                deformation = f[dset][:]
            # slclist attr will be is a list of strings: need them as datetimes

    except (IOError, OSError) as e:
        logger.error("Can't load %s in path %s: %s", filename, igram_path, e)
        return None, None
    try:
        slclist = load_slclist_from_h5(full_path, dset=dset)
    except Exception as e:
        logger.error(
            "Can't load slclist from %s in path %s: %s", filename, igram_path, e
        )
        slclist = None

    return slclist, deformation


def _load_deformation_npy(igram_path=None, filename=None, full_path=None, n=None):
    igram_path, filename, full_path = get_full_path(igram_path, filename, full_path)

    try:
        deformation = np.load(os.path.join(igram_path, filename))
        if n is not None:
            deformation = deformation[-n:]
        # slclist is a list of datetimes: encoding must be bytes
        slclist = np.load(
            os.path.join(igram_path, "slclist.npy"), encoding="bytes", allow_pickle=True
        )
    except (IOError, OSError):
        logger.error("%s or slclist.npy not found in path %s", filename, igram_path)
        return None, None

    return slclist, deformation


def load_datelist_from_h5(h5file, date_dset, dset=None, parse=True):
    from matplotlib.dates import num2date

    with h5py.File(h5file, "r") as f:
        if dset is None:
            datenums = f[date_dset][()]
        else:
            datenums = f[dset].attrs[date_dset][()]

    if parse:
        return num2date(datenums)
    else:
        return datenums


def load_slclist_from_h5(h5file, dset=None, parse=True, date_dset_name=SLCLIST_DSET):
    return load_datelist_from_h5(h5file, date_dset_name, dset, parse)


def load_ifglist_from_h5(h5file, dset=None, parse=True, date_dset_name=IFGLIST_DSET):
    return load_datelist_from_h5(h5file, date_dset_name, dset, parse)


def parse_slclist_strings(slc_str):
    """Parses a string, or list of strings, with YYYYmmdd as date"""
    # The re.search will find YYYYMMDD anywhere in string
    if isinstance(slc_str, str):
        match = re.search(r"\d{8}", slc_str)
        if not match:
            raise ValueError(f"{slc_str} does not contain date as YYYYMMDD")
        return _parse(match.group())
    else:
        # If it's an iterable of strings, run on each one
        return [parse_slclist_strings(s) for s in slc_str if s]


def parse_ifglist_strings(date_pairs):
    # If we passed filename YYYYmmdd_YYYYmmdd.int
    if isinstance(date_pairs, str):
        dates = re.findall(r"\d{8}", date_pairs)
        dates = list(sorted(set(dates)))
        if len(dates) != 2:
            raise ValueError("{} must contain 2 YYYYmmdd dates".format(date_pairs))
        return (_parse(dates[0]), _parse(dates[1]))
    else:
        return [parse_ifglist_strings(d) for d in date_pairs]


def load_slclist_from_nc(ncfile, dim="date", parse=True):
    import xarray as xr

    with xr.open_dataset(ncfile) as ds:
        dts = ds[dim].values.astype("datetime64[D]")
    if parse is True:
        return dts.tolist()
    else:
        return [d.item().strftime().strftime("%Y%m%d") for d in dts]


def _parse(datestr):
    return datetime.datetime.strptime(datestr, DATE_FMT).date()


def find_slcs(directory=".", ext=".geo", parse=True, filename=None):
    """Reads in the list of slc files used, in time order

    Can also pass a filename containing slc files as lines.

    Args:
        directory (str): path to the slclist file or directory
        ext (str): file extension when searching a directory
        parse (bool): output as parsed datetime tuples. False returns the filenames
        filename (string): name of a file with slc filenames

    Returns:
        list[date]: the parse dates of each slc used, in date order

    """
    if filename is not None:
        with open(filename) as f:
            slc_file_list = [
                line
                for line in f.read().splitlines()
                if not line.strip().startswith("#")
            ]
    else:
        slc_file_list = sorted(glob(os.path.join(directory, "*" + ext)))

    if not parse:
        return slc_file_list

    # Stripped of path for parser
    slclist = [os.path.split(fname)[1] for fname in slc_file_list]
    if not slclist:
        return []
    return parse_slclist_strings(slclist)


def find_igrams(directory=".", ext=".int", parse=True, filename=None, search_term=None):
    """Reads the list of igrams to return dates of images as a tuple

    Args:
        directory (str): path to the igram directory
        ext (str): file extension when searching a directory
        parse (bool): output as parsed datetime tuples. False returns the filenames
        filename (str): name of a file with filenames

    Returns:
        tuple(date, date) of (early, late) dates for all igrams (if parse=True)
            if parse=False: returns list[str], filenames of the igrams

    """
    if filename is not None:
        with open(filename) as f:
            igram_file_list = [
                line
                for line in f.read().splitlines()
                if not line.strip().startswith("#")
            ]
    elif search_term is not None:
        print("Searching for ifgs in:", os.path.join(directory, search_term))
        igram_file_list = sorted(glob(os.path.join(directory, search_term)))
    else:
        igram_file_list = sorted(glob(os.path.join(directory, "*" + ext)))

    if parse:
        igram_fnames = [os.path.split(f)[1] for f in igram_file_list]
        return parse_ifglist_strings(igram_fnames)
    else:
        return igram_file_list


find_ifgs = find_igrams


def load_dem_from_h5(h5file=None, dset="dem_rsc"):
    with h5py.File(h5file, "r") as f:
        return json.loads(f[dset][()])


def save_dem_to_h5(h5file, rsc_data, dset_name="dem_rsc", overwrite=True):
    if not check_dset(h5file, dset_name, overwrite):
        return

    with h5py.File(h5file, "a") as f:
        f[dset_name] = json.dumps(rsc_data)


def save_latlon_to_h5(h5file, lat=None, lon=None, rsc_data=None, overwrite=True):
    """Save the lat/lon information from a .rsc file as HDF5 scale datasets

    Args:
        fname (str): name of HDF5 file
        lat (ndarray): array of latitude points
        lon (ndarray): array of longitude points
        rsc_data (dict): data from an rsc file
            required if not passing lat/lon
    """
    from apertools.latlon import grid

    if not check_dset(h5file, "lat", overwrite) or not check_dset(
        h5file, "lon", overwrite
    ):
        return

    if lon is None or lat is None:
        lon, lat = grid(**rsc_data, sparse=True)
    with h5py.File(h5file, "a") as hf:
        ds = hf.create_dataset("lon", data=lon.ravel())
        hf["lon"].make_scale("longitude")
        ds.attrs["units"] = "degrees east"

        ds = hf.create_dataset("lat", data=lat.ravel())
        hf["lat"].make_scale("latitude")
        ds.attrs["units"] = "degrees north"


def attach_latlon(fname, dset, depth_dim=None):
    """Attach the lat/lon datasets (which are 'scales') to another dataset

    Args:
        fname (str): name of HDF5 file
        dset (str): Name of existing datasets in `fname` to attach the lat/lon scales to.
        depth_dim (str, optional): For a 3D dataset, attach a scale to the 3rd dimension
            If it doesn't exist, an index dimension, values 0,1,...len(dset), will be
            created with the name=`depth_dim`
    """
    with h5py.File(fname, "a") as hf:
        # Use ndim-2, ndim-1 in case it's 3D
        ndim = hf[dset].ndim
        # do i need these labels? eh, why not
        hf[dset].dims[ndim - 1].label = "lon"
        hf[dset].dims[ndim - 2].label = "lat"
        hf[dset].dims[ndim - 1].attach_scale(hf["lon"])
        hf[dset].dims[ndim - 2].attach_scale(hf["lat"])
        if depth_dim:
            if depth_dim not in hf:
                hf.create_dataset(depth_dim, data=np.arange(hf[dset].shape[0]))
                hf[depth_dim].make_scale(depth_dim)
            hf[dset].dims[0].attach_scale(hf[depth_dim])
            hf[dset].dims[0].label = depth_dim


def attach_latlon_2d(fname, dset, depth_dim=None):
    # Make dummy logical coordinates for the 2d lat/lon arrays
    with h5py.File(fname, "a") as hf:
        ndim = hf[dset].ndim
        # use the logical coords, since lat/lon are 2d
        hf[dset].dims[ndim - 2].attach_scale(hf["y"])
        hf[dset].dims[ndim - 1].attach_scale(hf["x"])
        # Used for xarray coordinate tracking
        coords = "lat lon"
        if depth_dim:
            if depth_dim not in hf:
                hf.create_dataset(depth_dim, data=np.arange(hf[dset].shape[0]))
                hf[depth_dim].make_scale(depth_dim)
            hf[dset].dims[0].attach_scale(hf[depth_dim])
            hf[dset].dims[0].label = depth_dim
            coords = depth_dim + " " + coords
        hf[dset].attrs["coordinates"] = coords


def create_2d_latlon_dataset(
    fname, array, lat, lon, dset_name=None, coord_3d_name=None, coord_3d_arr=None
):
    import xarray as xr

    if coord_3d_name is not None and coord_3d_arr is not None:
        if array.ndim != 3:
            raise ValueError("array must be 3D to pass coord_3d_name")
        coords = {
            coord_3d_name: coord_3d_arr,
            "lat": (["y", "x"], lat),
            "lon": (["y", "x"], lon),
        }
        dims = [coord_3d_name, "y", "x"]
    else:
        coords = {"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)}
        dims = ["y", "x"]

    da = xr.DataArray(array, coords=coords, dims=dims)
    ds = da.to_dataset(name=dset_name)
    ds.to_netcdf(fname, engine="h5netcdf", mode="a")


def load_rdr_latlon(geom_dir="geom_reference", lat_name="lat.rdr", lon_name="lon.rdr"):
    """Load the (lat, lon) 2d arrays corresponding to radar coorindates"""
    import rasterio as rio

    out_arrs = []
    for name in (lat_name, lon_name):
        with rio.open(os.path.join(geom_dir, name)) as src:
            out_arrs.append(src.read(1))
    return out_arrs


def save_latlon_2d_to_h5(
    h5file, lat=None, lon=None, geom_dir="geom_reference/", overwrite=True
):
    """Save the lat/lon information from a .rsc file as HDF5 scale datasets

    Args:
        fname (str): name of HDF5 file
        lat (ndarray): 2D array of latitude points
        lon (ndarray): 2D array of longitude points
    """
    if not check_dset(h5file, "lat", overwrite) or not check_dset(
        h5file, "lon", overwrite
    ):
        return
    if lon is None or lat is None:
        lon, lat = load_rdr_latlon(geom_dir=geom_dir)

    # Make dummy logical coordinates for the 2d lat/lon arrays
    xx, yy = np.arange(lon.shape[1]), np.arange(lon.shape[0])
    with h5py.File(h5file, "a") as hf:
        hf["x"] = xx
        hf["y"] = yy
        hf["x"].make_scale()
        hf["y"].make_scale()
        hf["lon"] = lon
        hf["lon"].attrs["units"] = "degrees east"
        hf["lon"].dims[0].attach_scale(hf["y"])
        hf["lon"].dims[1].attach_scale(hf["x"])
        hf["lat"] = lat
        hf["lat"].attrs["units"] = "degrees north"
        hf["lat"].dims[0].attach_scale(hf["y"])
        hf["lat"].dims[1].attach_scale(hf["x"])


def save_datelist_to_h5(
    date_dset,
    out_file=None,
    dset_name=None,
    date_list=None,
    overwrite=False,
):

    if dset_name is not None:
        if not check_dset(out_file, dset_name, overwrite, attr_name=date_dset):
            return date_list
    else:
        if not check_dset(out_file, date_dset, overwrite):
            return date_list

    with h5py.File(out_file, "a") as f:
        slc_datenums = slclist_to_num(date_list)
        if dset_name is not None:
            f[dset_name].attrs[date_dset] = slc_datenums
            f[dset_name].attrs[date_dset + "_units"] = get_datenum_units()
        else:
            f[date_dset] = slc_datenums
            f[date_dset].attrs["units"] = get_datenum_units()
    return date_list


def save_slclist_to_h5(
    igram_path=None,
    out_file=None,
    dset_name=None,
    slc_date_list=None,
    igram_ext=".int",
    overwrite=False,
    alt_name="date",
):
    if slc_date_list is None:
        slc_date_list, _ = load_slclist_ifglist(igram_path, igram_ext=igram_ext)
    dl = save_datelist_to_h5(
        SLCLIST_DSET, out_file, dset_name, slc_date_list, overwrite
    )
    if alt_name:
        with h5py.File(out_file, "a") as hf:
            if alt_name in hf:
                del hf[alt_name] 
            hf[alt_name] = hf[SLCLIST_DSET]

    return dl


def save_ifglist_to_h5(
    igram_path=None,
    dset_name=None,
    out_file=None,
    overwrite=False,
    igram_ext=".int",
    ifg_date_list=None,
):
    if ifg_date_list is None:
        _, ifg_date_list = load_slclist_ifglist(igram_path, igram_ext=igram_ext)
    return save_datelist_to_h5(
        IFGLIST_DSET, out_file, dset_name, ifg_date_list, overwrite
    )


def slclist_to_str(slc_date_list):
    return np.array([d.strftime(DATE_FMT) for d in slc_date_list]).astype("S")


def ifglist_to_str(ifg_date_list):
    """Date pairs to Nx2 numpy array of strings"""
    return np.array(
        [(a.strftime(DATE_FMT), b.strftime(DATE_FMT)) for a, b in ifg_date_list]
    ).astype("S")


def slclist_to_num(slc_date_list):
    """Convert list of slc dates into floats for 'days since 1970'
    Handles both strings and datetimes
    """
    return _date_list_to_num(slc_date_list)


def ifglist_to_num(ifg_date_list):
    """Convert list of ifg date pairs into floats for 'days since 1970'
    Handles both strings and datetimes
    """
    return _date_list_to_num(ifg_date_list)


def _date_list_to_num(ifg_date_list):
    """Convert list of dates, or list of date pairs, numpy array of floats
    for 'days since 1970'
    Handles both strings and datetimes
    """
    from matplotlib import dates

    arr = np.array(ifg_date_list)
    if isinstance(arr.ravel()[0], str):
        return dates.datestr2num(ifg_date_list)
    else:
        return dates.date2num(ifg_date_list)


def get_datenum_units():
    from matplotlib.dates import get_epoch

    # default: 'date.epoch': '1970-01-01T00:00:00',
    return "days since {}".format(get_epoch())


def ifglist_to_filenames(ifg_date_list, ext=".int"):
    """Convert date pairs to list of string filenames"""
    if isinstance(ifg_date_list[0], datetime.date):
        a, b = ifg_date_list
        return "{}_{}{ext}".format(a.strftime(DATE_FMT), b.strftime(DATE_FMT), ext=ext)
    else:
        return [ifglist_to_filenames(p, ext=ext) for p in ifg_date_list]


def load_slclist_ifglist(
    igram_dir=None,
    slc_dir=None,
    h5file=None,
    igram_ext=".int",
    parse=True,
):
    """Load the slc_date_list and ifg_date_list from a igram_dir with igrams

    if slc_dir is None, assumes that the slc files are one diretory up from the igrams
    """
    if h5file is not None:
        ifg_date_list = load_ifglist_from_h5(h5file, parse=parse)
        slc_date_list = load_slclist_from_h5(h5file, parse=parse)
    elif igram_dir is not None:
        ifg_date_list = find_igrams(directory=igram_dir, parse=parse, ext=igram_ext)
        if slc_dir is None:
            slc_dir = apertools.utils.get_parent_dir(igram_dir)
        slc_date_list = find_slcs(directory=slc_dir, parse=parse)

    return slc_date_list, ifg_date_list


def check_dset(h5file, dset_name, overwrite, attr_name=None):
    """Returns false if the dataset exists and overwrite is False

    If overwrite is set to true, will delete the dataset to make
    sure a new one can be created
    """
    with h5py.File(h5file, "a") as f:
        if attr_name is not None:
            if attr_name in f.get(dset_name, {}):
                msg = f"{dset_name}:{attr_name} already exists in {h5file}, "
                if overwrite:
                    del f[dset_name].attrs[attr_name]
                    msg += "overwrite true: Deleting."
                    logger.info(msg)
                else:
                    msg += "skipping"
                    logger.info(msg)
                    return False
        else:
            if dset_name in f:
                msg = f"{dset_name} already exists in {h5file}, "
                if overwrite:
                    msg += "overwrite true: Deleting."
                    logger.info(msg)
                    del f[dset_name]
                else:
                    msg += "skipping"
                    logger.info(msg)
                    return False

        return True


def load_mask(
    slc_date_list=None,
    perform_mask=True,
    deformation_filename=None,
    dset=None,
    mask_filename="masks.h5",
    directory=None,
):
    # TODO: Dedupe this from the insar one
    if not perform_mask:
        return np.ma.nomask

    if directory is not None:
        _, _, mask_full_path = get_full_path(
            directory=directory, filename=mask_filename
        )
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
            slc_date_list = load_slclist_from_h5(deformation_filename, dset=dset)

    # Get the indices of the mask layers that were used in the deformation stack
    all_slc_dates = load_slclist_from_h5(mask_full_path)
    if slc_date_list is None:
        used_bool_arr = np.full(len(all_slc_dates), True)
    else:
        used_bool_arr = np.array([g in slc_date_list for g in all_slc_dates])

    with h5py.File(mask_full_path) as f:
        # Maks a single mask image for any pixel that has a mask
        # Note: not using SLC_MASK_SUM_DSET since we may be sub selecting layers
        slc_dset = f[SLC_MASK_DSET]
        with slc_dset.astype(bool):
            stack_mask = np.sum(slc_dset[used_bool_arr, :, :], axis=0) > 0
        return stack_mask


def load_single_mask(
    int_date_string=None,
    date_pair=None,
    mask_filename=MASK_FILENAME,
    ifg_date_list=None,
):
    """Load one mask from the `mask_filename`

    Can either pass a tuple of Datetimes in date_pair, or a string like
    `20170101_20170104.int` or `20170101_20170303` to int_date_string
    """
    if ifg_date_list is None:
        ifg_date_list = load_ifglist_from_h5(mask_filename)

    if int_date_string is not None:
        # If the pass string with ., only take first part
        date_str_pair = int_date_string.split(".")[0].split("_")
        date_pair = parse_ifglist_strings([date_str_pair])[0]

    with h5py.File(mask_filename, "r") as f:
        idx = ifg_date_list.index(date_pair)
        dset = f[IFG_MASK_DSET]
        with dset.astype(bool):
            return dset[idx]


def load_isce_combined_mask(geom_dir=ISCE_GEOM_DIR, mask_files=ISCE_MASK_FILES):

    masks = []
    for mf in ISCE_MASK_FILES:
        m = load(os.path.join(geom_dir, mf), use_gdal=True, band=1).astype(bool)
        if "water" in mf.lower():
            # water has 1s on good pixels
            masks.append(~m)
        else:
            masks.append(m)
    return np.any(np.stack(masks), axis=0)


# ######### GDAL FUNCTIONS ##############


def save_as_geotiff(outfile=None, array=None, rsc_data=None, nodata=0.0):
    """Save an array to a GeoTIFF using gdal

    Ref: https://gdal.org/tutorials/raster_api_tut.html#using-create
    """
    from osgeo import gdal

    rows, cols = array.shape
    if rsc_data is not None and (
        rows != rsc_data["file_length"] or cols != rsc_data["width"]
    ):
        raise ValueError(
            "rsc_data ({}, {}) does not match array shape: ({}, {})".format(
                (rsc_data["file_length"], rsc_data["width"], rows, cols)
            )
        )

    driver = gdal.GetDriverByName("GTiff")

    gdal_dtype = numpy_to_gdal_type(array.dtype)
    out_raster = driver.Create(
        outfile, xsize=cols, ysize=rows, bands=1, eType=gdal_dtype
    )

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


def save_image_like(
    arr, outname, input_fname, driver=None, out_dtype=None, nodata=None
):
    """Writes out array `arr` to gdal-readable file `outname`
    using the georeferencing data from `input_fname`

    Used when saving a new image arr to the same lat/lon grid
    """
    import rasterio as rio

    # print(f"{arr=}, {outname=}, {input_fname=}")
    with rio.open(input_fname) as src:
        if (src.height, src.width) != arr.shape:
            raise ValueError(
                f"{input_fname} must be same size as arr to use georeference data"
            )

        with rio.open(
            outname,
            "w",
            driver=(driver or src.driver),
            height=arr.shape[0],
            width=arr.shape[1],
            transform=src.transform,
            count=1,
            dtype=(out_dtype or arr.dtype),
            crs=src.crs,
            nodata=(nodata or src.nodata),
        ) as dest:
            dest.write(arr, 1)


def save_vrt(
    filename=None,
    array=None,
    rows=None,
    cols=None,
    dtype=None,
    outfile=None,
    rsc_file=None,
    rsc_data=None,
    interleave=None,
    bands=[1],
    num_bands=None,
    relative=True,
    metadata_dict=None,
    metadata_domain=None,
    verbose=True,
    **kwargs,
):
    """Save a VRT corresponding to a raw raster file

    Args:
        TODO
        relative (bool): default True, save the VRT with a relative filename.
            False will save with the source filename as abspath

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
    # TODO: need to shift half pixel from RSC file to use GDAL conventions of
    # top left edge
    from osgeo import gdal

    gdal.UseExceptions()

    outfile = outfile or (filename + ".vrt")
    if outfile is None:
        raise ValueError("Need outfile or filename to save")

    if verbose:
        print(f"Saving {filename} to {outfile}, {relative = }")
    # Get geotransform and project based on rsc data, or existing GDAL info
    if rsc_data is None:
        if rsc_file is None:
            try:
                ds = gdal.Open(filename)
                geotrans = ds.GetGeoTransform()
                srs = ds.GetSpatialRef()
                band = ds.GetRasterBand(1)
                rows, cols = band.YSize, band.XSize
                band, ds = None, None
            except:
                print(
                    f"Warning: Cant get geotransform from {filename}, no .rsc file or data given"
                )
                geotrans = None
                srs = gdal.osr.SpatialReference()
                srs.SetWellKnownGeogCS("WGS84")
        else:
            rsc_data = load(rsc_file)
            geotrans = rsc_to_geotransform(rsc_data)
            srs = gdal.osr.SpatialReference()
            srs.SetWellKnownGeogCS("WGS84")

    if array is not None:
        dtype = array.dtype
        rows, cols = array.shape[-2:]

    if rsc_data is not None:
        rows, cols = _get_file_rows_cols(rsc_data=rsc_data)
        if array is not None:
            assert (rows, cols) == array.shape[-2:]

    if rows is None or cols is None:
        raise ValueError(
            f"Need to pass rows/col, rsc_file, or have {filename} be gdal-readable"
        )

    if dtype is None:
        dtype = _get_file_dtype(filename)
    gdal_dtype = numpy_to_gdal_type(dtype)

    bytes_per_pix = np.dtype(dtype).itemsize
    total_bytes = os.path.getsize(filename)
    if interleave is None or num_bands is None:
        interleave, num_bands = get_interleave(filename, num_bands=num_bands)
    if bands is None:
        # This will offset the start- only making the vrt point to phase
        bands = (
            [0, 1] if apertools.utils.get_file_ext(filename) in STACKED_FILES else [0]
        )

    assert rows == int(total_bytes / bytes_per_pix / cols / num_bands), (
        f"rows = total_bytes / bytes_per_pix / cols / num_bands : "
        f"{rows} = {total_bytes} / {bytes_per_pix} / {cols} / {num_bands} "
    )
    # assert total_bytes == bytes_per_pix * rows * cols

    vrt_driver = gdal.GetDriverByName("VRT")

    # out_raster = vrt_driver.Create(outfile, xsize=cols, ysize=rows, bands=1, eType=gdal_dtype)
    out_raster = vrt_driver.Create(outfile, xsize=cols, ysize=rows, bands=0)

    if geotrans is not None:
        out_raster.SetGeoTransform(geotrans)
    else:
        print("Warning: No GeoTransform could be made/set")

    out_raster.SetProjection(srs.ExportToWkt())
    if relative:
        rel = "1"
        source_filename = os.path.split(filename)[1]
    else:
        rel = "0"
        source_filename = os.path.abspath(filename)

    print(f"{bands = }")
    for band in bands:
        image_offset, pixel_offset, line_offset = get_offsets(
            dtype,
            interleave,
            band,
            cols,
            rows,
            num_bands,
        )
        options = [
            "subClass=VRTRawRasterBand",
            # split, since relative to file, so remove directory name
            f"SourceFilename={source_filename}",
            f"relativeToVRT={rel}",
            f"ImageOffset={image_offset}",
            f"PixelOffset={pixel_offset}",
            f"LineOffset={line_offset}",
            # 'ByteOrder=LSB'
        ]
        # print("gdal dtype", gdal_dtype, dtype)
        out_raster.AddBand(gdal_dtype, options)

    if metadata_dict is not None:
        out_raster.SetMetadata(metadata_dict, metadata_domain)
        # To write to a specific band:
        # band = out_raster.GetRasterBand(1)
        # band.SetMetadata(metadata_dict, metadata_domain)
    out_raster = None  # Force write

    # if geotrans is not None:
    # create_derived_band(outfile, rows, cols, geotrans, func="log10")
    # create_derived_band(outfile, rows, cols, geotrans, func="phase")
    return outfile


def shift_by_pixel(in_f, out_f, full_pixel=True, down_right=False):
    """Shift a raster up and to the left by 1/2 (or 1 if `full_pixel`=True)

    if `down_right` is True, shifts in the opposite direction.

    Can fix problem of pixel center vs pixel edge convention differences
    """
    import copy
    import rasterio as rio

    denom = 1 if full_pixel else 2
    if down_right:
        denom *= -1
    with rio.open(in_f) as src:
        meta2 = copy.deepcopy(src.meta)
        tsl = list(src.meta["transform"])
        tsl[2] -= tsl[0] / denom
        tsl[5] -= tsl[4] / denom
        a2 = rio.transform.Affine(*tsl[:6])
        meta2["transform"] = a2
        with rio.open(out_f, "w", **meta2) as dst:
            dst.write(src.read())


def create_derived_band(
    src_filename, outfile=None, src_dtype="CFloat32", desc=None, func="log10"
):
    from osgeo import gdal

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
    """Returns band interleave format, and number of bands

    Band interleaved by line (BIL), band interleaved by pixel (BIP), and band sequential (BSQ)
    https://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/bil-bip-and-bsq-raster-files.htm
    """
    if num_bands == 1:
        # 1 band is always same: its just all pixels in a row
        return "BIP", 1

    ext = apertools.utils.get_file_ext(filename)
    if ext in BIL_FILES:
        interleave, num_bands = "BIL", 2
    # TODO: the .amp files are actually BIP with 2 bands...
    elif ext in BIP_FILES:
        interleave, num_bands = "BIP", 1
    else:
        raise ValueError(
            "Unknown band interleave format (BIP/BIL) for {}".format(filename)
        )
    return interleave, num_bands


def get_offsets(dtype, interleave, band, width, length, num_bands=1):
    """
    From ISCE Image.py
    """
    bytes_per_pix = np.dtype(dtype).itemsize
    # In this single-band case, all choices are the same
    if band == 0 and num_bands == 1:
        return (
            width * bytes_per_pix,  # ImageOffset
            bytes_per_pix,  # PixelOffset
            width * bytes_per_pix,  # LineOffset
        )
    # otherwise, get the specific interleave options
    if interleave == "BIL":
        return (
            band * width * bytes_per_pix,  # ImageOffset
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


def rsc_to_geotransform(rsc_data, half_shift=True):
    """Convert the data in an .rsc file to a 6 element geotransform for GDAL

    See here for geotransform info
    https://gdal.org/user/raster_data_model.html#affine-geotransform
    NOTE: `half_shift` argument is because gdal standard is to
    reference a pixel by its top left corner,
    while often the .rsc for SAR focusing is using the middle of a pixel.

    Xgeo = GT(0) + Xpixel*GT(1) + Yline*GT(2)
    Ygeo = GT(3) + Xpixel*GT(4) + Yline*GT(5)

    So for us, this means we have
    X0 = trans[0] + .5*trans[1] + (.5*trans[2])
    Y0 = trans[3] + (.5*trans[4]) + .5*trans[5]
    where trans[2], trans[4] are 0s for north-up rasters
    """

    x_step = rsc_data["x_step"]
    y_step = rsc_data["y_step"]
    X0 = rsc_data["x_first"]
    Y0 = rsc_data["y_first"]
    if half_shift:
        X0 -= 0.5 * x_step
        Y0 -= 0.5 * y_step
    return (X0, x_step, 0.0, Y0, 0.0, y_step)


def set_unit(filename, unit="cm", band=None):
    from osgeo import gdalconst
    from osgeo import gdal

    ds = gdal.Open(filename, gdalconst.GA_Update)
    if band is not None:
        b = ds.GetRasterBand(band)
        b.SetUnitType(unit)
        b = None
    else:
        for band in range(1, ds.RasterCount + 1):
            b = ds.GetRasterBand(band)
            b.SetUnitType(unit)
            b = None
    ds = None


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
    rgbs = np.flipud(
        rgbs
    )  # flip up-down so that orange is in the later half (positive)
    return rgbs


def make_cmy_colortable():
    from osgeo import gdal

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


def find_looks_taken(
    igram_path,
    slc_path=None,
    igram_dem_file="dem.rsc",
    slc_dem_file="elevation.dem.rsc",
):
    """Calculates how many looks from slc files to .int files"""
    if slc_path is None:
        slc_path = os.path.dirname(os.path.abspath(igram_path))

    slc_dem_rsc = load_dem_rsc(os.path.join(slc_path, slc_dem_file))

    igram_dem_rsc = load_dem_rsc(os.path.join(igram_path, igram_dem_file))

    row_looks = slc_dem_rsc["file_length"] // igram_dem_rsc["file_length"]
    col_looks = slc_dem_rsc["width"] // igram_dem_rsc["width"]
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
    rsc_dict = load_dem_rsc(filename=rsc_filename)
    default_spacing = 1.0 / 3600  # NASA SRTM uses 3600 pixels for 1 degree, or 30 m
    x_spacing = abs(rsc_dict["x_step"])
    y_spacing = abs(rsc_dict["y_step"])
    return default_spacing / x_spacing, default_spacing / y_spacing


def save_slc_amp_stack(
    directory=".", ext=".slc", dtype="float32", outname="slc_stack.nc"
):
    """Save a bunch of SLCs into one NetCDF stack"""
    import xarray as xr

    fnames = glob("*" + ext)
    date_list = find_slcs(directory=directory, ext=ext, parse=True)
    date_list = np.array(date_list, dtype="datetime64[D]")
    with xr.open_rasterio(fnames[0]) as ds1:
        _, rows, cols = ds1.shape

        data = xr.DataArray(
            np.empty((len(fnames), rows, cols), dtype=dtype),
            dims=("date", "lat", "lon"),
            coords={
                "date": date_list,
                "lat": ds1.coords["y"].data,
                "lon": ds1.coords["x"].data,
            },
            attrs=ds1.attrs,
        )

    for idx, f in enumerate(fnames):
        with xr.open_rasterio(f) as ds:
            data[idx, :, :] = np.abs(ds[0])
    data.to_netcdf(outname)
    return data


def save_east_up_mat(east_up_fname, outname=None, units="cm"):
    """From los.solve_east_up results, save a new .mat file for MATLAB use"""
    import rasterio as rio
    import apertools.latlon as latlon
    from scipy.io import savemat

    if outname is None:
        outname = east_up_fname + ".mat"

    with rio.open(east_up_fname) as src:
        east = src.read(1)
        up = src.read(2)
        lons, lats = latlon.grid(fname=east_up_fname, sparse=True)
    savemat(
        outname,
        dict(
            lons=lons,
            lats=lats,
            east=east,
            up=up,
            units=units,
        ),
    )


def make_unw_vrt(unw_filelist=None, directory=None, output="unw_stack.vrt", ext=".unw"):
    from osgeo import gdal

    if unw_filelist is None:
        unw_filelist = glob(os.path.join(directory, "*" + ext))

    gdal.BuildVRT(output, unw_filelist, separate=True, srcNodata="nan 0.0")
    # But we want the 2nd band (not an option on build for some reason)
    with fileinput.FileInput(output, inplace=True) as f:
        for line in f:
            print(
                line.replace(
                    "<SourceBand>1</SourceBand>", "<SourceBand>2</SourceBand>"
                ),
                end="",
            )
