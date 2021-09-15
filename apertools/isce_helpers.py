# adapted from components/isceobj/StripmapProc/runInterferogram.py
import os
import glob
import datetime
import subprocess
import numpy as np

# from osgeo import gdal
import rasterio as rio
from tqdm import tqdm
from copy import deepcopy
import h5py
from . import sario

import isce  # noqa
import isceobj
from components.stdproc.stdproc import crossmul
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from mroipac.looks.Looks import Looks
from mroipac.filter.Filter import Filter


from apertools.log import get_log

logger = get_log()


def multilook(infile, outname=None, alks=5, rlks=15):
    """
    Take looks.
    """

    logger.info("Multilooking {} ...".format(infile))

    inimg = isceobj.createImage()
    inimg.load(infile + ".xml")

    if outname is None:
        spl = os.path.splitext(inimg.filename)
        ext = ".{0}alks_{1}rlks".format(alks, rlks)
        outname = spl[0] + ext + spl[1]

    lkObj = Looks()
    lkObj.setDownLooks(alks)
    lkObj.setAcrossLooks(rlks)
    lkObj.setInputImage(inimg)
    lkObj.setOutputFilename(outname)
    lkObj.looks()

    return outname


def generateIgram(slcFile1, slcFile2, ifgFile, azLooks, rgLooks, compute_cor=True):
    imageSlc1 = isceobj.createImage()
    f1 = slcFile1 + ".xml" if not slcFile1.endswith(".xml") else slcFile1
    print(f"Loading {f1}")
    imageSlc1.load(f1)

    imageSlc2 = isceobj.createImage()
    f2 = slcFile2 + ".xml" if not slcFile2.endswith(".xml") else slcFile2
    print(f"Loading {f2}")
    imageSlc2.load(f2)

    objSlc1 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc1, objSlc1)
    objSlc1.setAccessMode("read")
    objSlc1.createImage()

    objSlc2 = isceobj.createSlcImage()
    IU.copyAttributes(imageSlc2, objSlc2)
    objSlc2.setAccessMode("read")
    objSlc2.createImage()

    slcWidth = imageSlc1.getWidth()

    intWidth = slcWidth // rgLooks

    lines = min(imageSlc1.getLength(), imageSlc2.getLength())

    if ".flat" in ifgFile:
        resampAmp = ifgFile.replace(".flat", ".amp")
    elif ".int" in ifgFile:
        resampAmp = ifgFile.replace(".int", ".amp")
    else:
        resampAmp = ifgFile + ".amp"

    resampInt = ifgFile

    objInt = isceobj.createIntImage()
    objInt.setFilename(resampInt)
    objInt.setWidth(intWidth)
    imageInt = isceobj.createIntImage()
    IU.copyAttributes(objInt, imageInt)
    objInt.setAccessMode("write")
    objInt.createImage()

    objAmp = isceobj.createAmpImage()
    objAmp.setFilename(resampAmp)
    objAmp.setWidth(intWidth)
    imageAmp = isceobj.createAmpImage()
    IU.copyAttributes(objAmp, imageAmp)
    objAmp.setAccessMode("write")
    objAmp.createImage()

    objCrossmul = crossmul.createcrossmul()
    objCrossmul.width = slcWidth
    objCrossmul.length = lines
    objCrossmul.LooksDown = azLooks
    objCrossmul.LooksAcross = rgLooks

    objCrossmul.crossmul(objSlc1, objSlc2, objInt, objAmp)

    for obj in [objInt, objAmp, objSlc1, objSlc2]:
        obj.finalizeImage()

    if compute_cor:
        # Compute the multilooked version of correlation (much quicker than isce's full size)
        with rio.open(resampInt) as src:
            ifg = src.read(1)
        with rio.open(resampAmp) as src:
            amp1, amp2 = src.read()

        cor_filename = resampAmp.replace(".amp", ".cor")
        # Make the isce headers
        create_cor_image(cor_filename, ifg.shape, access_mode="write")
        # calulate and save (not sure how to just save an array in isce)
        cor = np.abs(ifg) / np.sqrt(amp1 ** 2 * amp2 ** 2)
        sario.save(cor_filename, np.stack((amp1 * amp2, cor)))

    return imageInt, imageAmp


def _create_isce_image(
    filename, shape, image_class=None, data_type="FLOAT", bands=1, access_mode="write"
):

    length, width = shape
    filename = os.path.abspath(filename)

    if image_class is None:
        image = isceobj.createImage()
        image.dataType = data_type
        image.bands = bands
    elif image_class == "UnwImage":
        imgFunc = getattr(isceobj.Image, "create" + image_class)
        image = imgFunc()
    else:
        imgFunc = getattr(isceobj, "create" + image_class)
        image = imgFunc()

    image.setFilename(filename)
    image.setWidth(width)
    image.setLength(length)
    image.setAccessMode(access_mode)
    if not os.path.exists(filename):
        image.createImage()
        image.renderHdr()
        image.finalizeImage()
    else:
        image.renderHdr()
    return image


def create_cor_image(cor_filename, shape, bands=2, access_mode="read"):
    if bands == 1:
        return _create_isce_image(
            cor_filename,
            shape,
            data_type="FLOAT",
            bands=bands,
            access_mode=access_mode,
        )
    else:
        return _create_isce_image(
            cor_filename, shape, image_class="OffsetImage", access_mode=access_mode
        )


def _try_get_shape(filename, extra_exts=[]):
    shape = None
    ext = os.path.splitext(filename)[1]
    for newext in [ext] + extra_exts:
        try:
            with rio.open(filename.replace(ext, newext)) as src:
                shape = src.shape
        except rio.RasterioIOError:
            pass
    if shape is None:
        raise ValueError("Cant open %s, need to pass `shape`" % filename)
    else:
        return shape


def create_unw_image(filename, shape=None, access_mode="read"):
    if shape is None:
        shape = _try_get_shape(filename, extra_exts=[".int"])
    return _create_isce_image(filename, shape, "UnwImage", access_mode=access_mode)


def create_int_image(filename, shape=None, access_mode="read"):
    if shape is None:
        shape = _try_get_shape(filename)
    return _create_isce_image(filename, shape, "IntImage", access_mode=access_mode)


def filter_ifg(ifgFilename, outname=None, filterStrength=0.5):
    ifgDirname = os.path.dirname(ifgFilename)
    filename_only = os.path.split(ifgFilename)[1]
    if outname is None:
        outname = os.path.join(ifgDirname, "filt_" + filename_only)

    img1 = isceobj.createImage()
    img1.load(ifgFilename + ".xml")
    widthInt = img1.getWidth()

    intImage = isceobj.createIntImage()
    intImage.setFilename(ifgFilename)
    intImage.setWidth(widthInt)
    intImage.setAccessMode("read")
    intImage.createImage()

    # Create the filtered interferogram
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(outname)
    filtImage.setWidth(widthInt)
    filtImage.setAccessMode("write")
    filtImage.createImage()

    objFilter = Filter()
    objFilter.wireInputPort(name="interferogram", object=intImage)
    objFilter.wireOutputPort(name="filtered interferogram", object=filtImage)

    objFilter.goldsteinWerner(alpha=filterStrength)

    intImage.finalizeImage()
    filtImage.finalizeImage()


def get_square_pixel_looks(frame, posting, azlooks=None, rglooks=None):
    """
    Compute relevant number of looks.
    """
    from isceobj.Planet.Planet import Planet
    from isceobj.Constants import SPEED_OF_LIGHT

    azFinal = None
    rgFinal = None

    if azlooks is not None:
        azFinal = azlooks

    if rglooks is not None:
        rgFinal = rglooks

    if (azFinal is not None) and (rgFinal is not None):
        return (azFinal, rgFinal)

    if posting is None:
        raise Exception(
            "Input posting is none. Either specify (azlooks, rglooks) or posting in input file"
        )

    elp = Planet(pname="Earth").ellipsoid

    ####First determine azimuth looks
    tmid = frame.sensingMid
    sv = frame.orbit.interpolateOrbit(tmid, method="hermite")  # .getPosition()
    llh = elp.xyz_to_llh(sv.getPosition())

    if azFinal is None:
        hdg = frame.orbit.getENUHeading(tmid)
        elp.setSCH(llh[0], llh[1], hdg)
        sch, vsch = elp.xyzdot_to_schdot(sv.getPosition(), sv.getVelocity())
        azFinal = max(int(np.round(posting * frame.PRF / vsch[0])), 1)

    if rgFinal is None:
        pulseLength = frame.instrument.pulseLength
        chirpSlope = frame.instrument.chirpSlope

        # Range Bandwidth
        rBW = np.abs(chirpSlope) * pulseLength

        # Slant Range resolution
        rgres = abs(SPEED_OF_LIGHT / (2.0 * rBW))

        r0 = frame.startingRange
        rmax = frame.getFarRange()
        rng = (r0 + rmax) / 2

        Re = elp.pegRadCur
        H = sch[2]
        cos_beta_e = (Re ** 2 + (Re + H) ** 2 - rng ** 2) / (2 * Re * (Re + H))
        sin_bet_e = np.sqrt(1 - cos_beta_e ** 2)
        sin_theta_i = sin_bet_e * (Re + H) / rng
        print(
            "incidence angle at the middle of the swath: ",
            np.arcsin(sin_theta_i) * 180.0 / np.pi,
        )
        groundRangeRes = rgres / sin_theta_i
        print("Ground range resolution at the middle of the swath: ", groundRangeRes)
        rgFinal = max(int(np.round(posting / groundRangeRes)), 1)

    return azFinal, rgFinal


# def get_uavsar_velocity():
#     mdd = dict(metadata)
#     elp.setSCH(mdd['Peg Latitude'], mdd['Peg Longitude'], mdd['Peg Heading'])
#     scale = (elp.pegRadCur + mdd['Average Altitude']) / elp.pegRadCur
#     scale * mdd['Azimuth Spacing'] / mdd['Average Pulse Repetition Interval']


def save_uavsar_hdf5_slc(hdf5_file, output, frequency="A", polarization="HH"):
    with h5py.File(hdf5_file, "r") as hf:
        ds = hf["/science/LSAR/SLC/swaths/frequency" + frequency + "/" + polarization]
        with ds.astype(np.complex64):
            with open(output, "wb") as fout:
                ds[:].tofile(fout)


def extract_uavsar_hdf5_orbit(hdf5_file, output, frequency="A", polarization="HH"):
    with h5py.File(hdf5_file, "r") as hf:
        referenceUTC = (
            hf["/science/LSAR/SLC/swaths/zeroDopplerTime"]
            .attrs["units"]
            .decode("utf-8")
        )
        referenceUTC = referenceUTC.replace("seconds since ", "")
        format_str = "%Y-%m-%d %H:%M:%S"
        if "." in referenceUTC:
            format_str += ".%f"
        t0 = datetime.datetime.strptime(referenceUTC, format_str)
        # given as seconds since some epoch, t0
        t_offset_arr = hf["/science/LSAR/SLC/metadata/orbit/time"][:]
        t_arr = [t0 + datetime.timedelta(seconds=t) for t in t_offset_arr]

        position = hf["/science/LSAR/SLC/metadata/orbit/position"][:]
        velocity = hf["/science/LSAR/SLC/metadata/orbit/velocity"][:]
        return t_arr, position, velocity


def create_dem_header(dem_file, rsc_file=None, datum="EGM96"):
    from isceobj.Image import createDemImage

    if rsc_file is None:
        rsc_file = dem_file + ".rsc"

    rsc_dict = sario.load(rsc_file)
    width, length = rsc_dict["width"], rsc_dict["file_length"]

    demImage = createDemImage()
    demImage.initImage(dem_file, "read", width)
    demImage.dataType = "BYTE"

    dictProp = {
        "REFERENCE": datum,
        "Coordinate1": {
            "size": width,
            "startingValue": rsc_dict["x_first"],
            "delta": rsc_dict["x_step"],
        },
        "Coordinate2": {
            "size": length,
            "startingValue": rsc_dict["y_first"],
            "delta": rsc_dict["y_step"],
        },
        "FILE_NAME": dem_file,
    }
    # no need to pass the dictionaryOfFacilities since init will use the default one
    demImage.init(dictProp)
    demImage.renderHdr()


def create_unfiltered_cor_for_project(
    project_dir, search_term="Igrams/**/2*.int", verbose=False
):

    ifglist = sario.find_ifgs(
        directory=project_dir, search_term=search_term, parse=False
    )

    for f1 in tqdm(ifglist):
        create_cor_from_int_amp(f1, verbose=verbose)


def create_unfiltered_phsig_for_project(project_dir, search_term="Igrams/**/2*.int"):
    from . import utils

    ifglist = sario.find_ifgs(
        directory=project_dir, search_term=search_term, parse=False
    )

    for f1 in tqdm(ifglist):
        dirname, fname = os.path.split(f1)
        with utils.chdir_then_revert(dirname):
            create_phsig(fname)


def create_cor_from_int_amp(fname, verbose=False, mask=None):
    with rio.open(fname) as src1:
        shape = src1.shape

    a1 = fname.replace(".int", ".amp")
    cor_filename = a1.replace(".amp", ".cor")
    if verbose:
        tqdm.write("Saving cor for %s" % cor_filename)

    with rio.open(fname) as src1, rio.open(a1) as src2:
        ifg = src1.read(1)
        amp1, amp2 = src2.read()

        # For output: copy all metadata, but change band count
        meta = deepcopy(src2.meta)
        meta["count"] = 1

    cor = np.abs(ifg) / (amp1 * amp2 + 1e-7)
    if mask is not None:
        cor[mask] = 0
    # calulate and save (not sure how to just save an array in isce)
    with rio.open(cor_filename, "w", **meta) as dst:
        dst.write(cor, 1)
    create_cor_image(cor_filename, shape, bands=1, access_mode="write")


def create_phsig(ifg_file, cor_file=None):
    from mroipac.icu.Icu import Icu

    logger.info("Estimating spatial coherence based phase sigma")
    if cor_file is None:
        ext = os.path.splitext(ifg_file)[1]
        cor_file = ifg_file.replace(ext, ".cor")
        logger.info("writing output to %s", cor_file)

    # Create phase sigma correlation file here
    intImage = isceobj.createIntImage()
    intImage.load(ifg_file.replace(".xml", "") + ".xml")
    intImage.setAccessMode("read")
    intImage.createImage()

    phsigImage = isceobj.createImage()
    phsigImage.dataType = "FLOAT"
    phsigImage.bands = 1
    phsigImage.setWidth(intImage.getWidth())
    phsigImage.setFilename(cor_file)
    phsigImage.setAccessMode("write")
    phsigImage.createImage()

    icuObj = Icu(name="filter_icu")
    icuObj.configure()
    icuObj.unwrappingFlag = False
    icuObj.useAmplitudeFlag = False

    icuObj.icu(intImage=intImage, phsigImage=phsigImage)
    phsigImage.renderHdr()

    intImage.finalizeImage()
    phsigImage.finalizeImage()


def multilook_configs(
    project_dir=".", looks=(15, 9), max_temp=None, crossmul_only=False
):
    from . import utils

    if not max_temp:
        max_temp = 100000

    # TODO: fix
    os.chdir(project_dir)

    row_looks, col_looks = looks
    new_igrams_dir = f"Igrams_{row_looks}_{col_looks}"
    utils.mkdir_p(new_igrams_dir)
    new_configs_dir = f"configs_{row_looks}_{col_looks}"
    utils.mkdir_p(new_configs_dir)

    new_run_file = f"run_igram_{row_looks}_{col_looks}.sh"
    with open(new_run_file, "w") as f_run:
        f_run.write("set -e\n")
    stripmap_line = "stripmapWrapper.py -c {}"

    for fname in glob.glob("configs/config_igram_*"):
        ifg = sario.parse_ifglist_strings(fname)
        baseline = (ifg[1] - ifg[0]).days
        if baseline > max_temp:
            continue
        with open(fname) as f:
            cfg_lines = f.read().splitlines()

        out_lines = []
        for line in cfg_lines:
            # If we only want to crossmul, skip the rest
            if "Function-2" in line and crossmul_only:
                break
            if "alks" in line:  # azimuth -> row
                out_lines.append(f"alks : {row_looks}")
            elif "rlks" in line:  # range -> col
                out_lines.append(f"rlks : {col_looks}")
            elif "Igrams" in line:
                out_lines.append(line.replace("Igrams", new_igrams_dir))
            else:
                out_lines.append(line)

        out_fname = os.path.join(new_configs_dir, os.path.split(fname)[1])
        with open(out_fname, "w") as fout:
            fout.write("\n".join(out_lines))
            fout.write("\n")
        with open(new_run_file, "a") as f_run:
            f_run.write(stripmap_line.format(os.path.abspath(out_fname)) + "\n")

    return new_run_file, new_configs_dir


def multilook_geom(looks=(15, 9), geom_dir="geom_reference", overwrite=False):
    """Redo the geom_reference files with the extra multilook factor for unwrapping"""
    from . import utils

    geom_files = glob.glob(os.path.join(geom_dir, "*.rdr"))
    glooks = utils.get_looks_rdr(geom_files[0])

    with rio.open(geom_files[0]) as src:
        in_rows, in_cols = src.shape[-2:]
    extra_row_looks = looks[0] // glooks[0]
    extra_col_looks = looks[1] // glooks[1]
    out_rows = in_rows // extra_row_looks
    out_cols = in_cols // extra_col_looks

    geom_dir_new = geom_dir.rstrip("/") + f"_{looks[0]}_{looks[1]}"
    utils.mkdir_p(geom_dir_new)

    for f in tqdm(geom_files):
        try:
            src = rio.open(f)
            src.close()
        except rio.errors.RasterioIOError:
            logger.warning("Cant open %s, skipping", f)
            continue
        f_out = f.replace(geom_dir, geom_dir_new)
        if os.path.exists(f_out) and not overwrite:
            continue
        cmd = f"gdal_translate -of ISCE -outsize {out_cols} {out_rows} {f} {f_out}"
        tqdm.write(cmd)
        subprocess.check_call(cmd, shell=True)
