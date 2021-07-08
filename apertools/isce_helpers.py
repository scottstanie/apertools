# adapted from components/isceobj/StripmapProc/runInterferogram.py
import os
import numpy as np
from osgeo import gdal
from . import sario

import isce
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
        make_cor_image(cor_filename, ifg.shape)
        # calulate and save (not sure how to just save an array in isce)
        cor = np.abs(ifg) / np.sqrt(amp1 ** 2 * amp2 ** 2)
        sario.save(cor_filename, np.stack((amp1 * amp2, cor)))

    return imageInt, imageAmp


def make_cor_image(cor_filename, shape):
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


def make_unw_image(filename, shape):
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


def extractDoppler(metadata, frame):
    # Recast the Near, Mid, and Far Reskew Doppler values
    # into three RDF records because they were not parsed
    # correctly by the RDF parser; it was parsed as a string.
    # Use the RDF parser on the individual Doppler values to
    # do the unit conversion properly.

    from iscesys.Parsers.rdf import iRDF

    # The units, and values parsed from the metadataFile
    key = "Reskew Doppler Near Mid Far"
    u = metadata.data[key].units.split(",")
    v = map(float, metadata.data[key].value.split())
    k = ["Reskew Doppler " + x for x in ("Near", "Mid", "Far")]

    # Use the interactive RDF accumulator to create an RDF object
    # for the near, mid, and far Doppler values
    dop = iRDF.RDFAccumulator()
    for z in zip(k, u, v):
        dop("%s (%s) = %f" % z)
    dopplerVals = {}
    for r in dop.record_list:
        dopplerVals[r.key.split()[-1]] = r.field.value

    # Quadratic model using Near, Mid, Far range doppler values
    # UAVSAR has a subroutine to compute doppler values at each pixel
    # that should be used instead.
    frame = frame
    instrument = frame.getInstrument()
    width = frame.getNumberOfSamples()
    nearRangeBin = 0.0
    midRangeBin = float(int((width - 1.0) / 2.0))
    farRangeBin = width - 1.0

    A = np.matrix(
        [
            [1.0, nearRangeBin, nearRangeBin ** 2],
            [1.0, midRangeBin, midRangeBin ** 2],
            [1.0, farRangeBin, farRangeBin ** 2],
        ]
    )
    d = np.matrix(
        [dopplerVals["Near"], dopplerVals["Mid"], dopplerVals["Far"]]
    ).transpose()
    coefs = (np.linalg.inv(A) * d).transpose().tolist()[0]
    prf = instrument.getPulseRepetitionFrequency()
    coefs_norm = {"a": coefs[0] / prf, "b": coefs[1] / prf, "c": coefs[2] / prf}

    return coefs_norm, coeffs