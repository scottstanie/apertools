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
    outImage.setAccessMode('read')
    outImage.renderHdr()
    outImage.renderVRT()