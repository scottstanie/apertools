# adapted from components/isceobj/StripmapProc/runInterferogram.py
import isce
import isceobj
from components.stdproc.stdproc import crossmul
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from mroipac.looks.Looks import Looks
import os

# from osgeo import gdal
# import numpy as np

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


def generateIgram(imageSlc1, imageSlc2, resampName, azLooks, rgLooks):
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

    return imageInt, imageAmp
