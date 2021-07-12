# adapted from components/isceobj/StripmapProc/runInterferogram.py
import os
import datetime
import numpy as np
from osgeo import gdal
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


def generateIgram(file1, file2, resampName, azLooks, rgLooks, compute_cor=True):
    imageSlc1 = isceobj.createImage()
    f1 = file1 + ".xml" if not file1.endswith(".xml") else file1
    print(f"Loading {f1}")
    imageSlc1.load(f1)

    imageSlc2 = isceobj.createImage()
    f2 = file2 + ".xml" if not file2.endswith(".xml") else file2
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

    if ".flat" in resampName:
        resampAmp = resampName.replace(".flat", ".amp")
    elif ".int" in resampName:
        resampAmp = resampName.replace(".int", ".amp")
    else:
        resampAmp = resampName + ".amp"

    resampInt = resampName

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
        ds = gdal.Open(resampInt)
        ifg = ds.ReadAsArray()
        ds = None
        ds = gdal.Open(resampAmp)
        amp1, amp2 = ds.ReadAsArray()
        ds = None

        cor_filename = resampAmp.replace(".amp", ".cor")
        # Make the isce headers
        create_cor_image(cor_filename, ifg.shape)
        # calulate and save (not sure how to just save an array in isce)
        cor = np.abs(ifg) / np.sqrt(amp1 ** 2 * amp2 ** 2)
        sario.save(cor_filename, np.stack((amp1 * amp2, cor)))

    return imageInt, imageAmp


def create_cor_image(cor_filename, shape):
    length, width = shape
    cohImage = isceobj.createOffsetImage()
    cohImage.setFilename(cor_filename)
    cohImage.setWidth(width)
    cohImage.setLength(length)
    cohImage.setAccessMode("write")
    cohImage.createImage()
    cohImage.renderHdr()
    cohImage.finalizeImage()


def filter_ifg(ifgFilename, filterStrength=0.5):
    ifgDirname = os.path.dirname(ifgFilename)
    filename_only = os.path.split(ifgFilename)[1]

    img1 = isceobj.createImage()
    img1.load(ifgFilename + ".xml")
    widthInt = img1.getWidth()

    intImage = isceobj.createIntImage()
    intImage.setFilename(ifgFilename)
    intImage.setWidth(widthInt)
    intImage.setAccessMode("read")
    intImage.createImage()

    # Create the filtered interferogram
    filtIntFilename = os.path.join(ifgDirname, "filt_" + filename_only)
    filtImage = isceobj.createIntImage()
    filtImage.setFilename(filtIntFilename)
    filtImage.setWidth(widthInt)
    filtImage.setAccessMode("write")
    filtImage.createImage()

    objFilter = Filter()
    objFilter.wireInputPort(name="interferogram", object=intImage)
    objFilter.wireOutputPort(name="filtered interferogram", object=filtImage)

    objFilter.goldsteinWerner(alpha=filterStrength)

    intImage.finalizeImage()
    filtImage.finalizeImage()


def create_unw_image(filename, shape):
    length, width = shape
    outImage = isceobj.Image.createUnwImage()
    outImage.setFilename(filename)
    outImage.setWidth(width)
    outImage.setAccessMode("read")
    outImage.renderHdr()
    outImage.renderVRT()


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