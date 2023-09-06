#!/usr/bin/env python
import argparse
import os
import re
from pathlib import Path

from osgeo import gdal

SENTINEL_WAVELENGTH = 0.05546576
ALOS_WAVELENGTH = 299792458 / 1.270e9  # 0.23605705354330708


def get_cli_args():
    parser = argparse.ArgumentParser(description="Convert SLC stack to single VRT")
    parser.add_argument(
        "--in-vrts",
        nargs="*",
        help="Merged directory of tops stack generation",
    )
    parser.add_argument(
        "--in-vrts-file",
        type=str,
        help="Alternative to --in-vrts: filename with list of SLC files to use",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="stack",
        help="Directory where the vrt stack will be stored (default is %(default)s)",
    )
    parser.add_argument(
        "--out-vrt-name",
        type=str,
        default="slcs_base.vrt",
        help="Name of output SLC containing all images (defaul = %(default)s)",
    )
    parser.add_argument(
        "--subset-bbox",
        type=int,
        nargs=4,
        metavar=("left", "bottom", "right", "top"),
        default=None,
        help="Bounding box to subset the stack to (default is full stack)",
    )
    parser.add_argument(
        "--target-extent",
        type=int,
        nargs=4,
        metavar=("xmin", "ymin", "xmax", "ymax"),
        default=None,
        help="Target extent (like GDAL's `-te` option), alternative to subset the stack.",
    )
    args = parser.parse_args()
    return args


def create_vrt_stack(file_list, outfile="slcs_base.vrt"):
    # Use the first file in the stack to get size, transform info
    ds = gdal.Open(file_list[0])
    geotrans = ds.GetGeoTransform()
    proj = ds.GetProjection()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    ds = None

    # Start with the empty VRT, and add 1 band at a time per file
    vrt_driver = gdal.GetDriverByName("VRT")
    out_raster = vrt_driver.Create(outfile, xsize=xsize, ysize=ysize, bands=0)
    out_raster.SetGeoTransform(geotrans)
    out_raster.SetProjection(proj)

    # Options for all VRT bands:
    gdal_dtype = gdal.GetDataTypeByName("CFloat32")
    bytes_per_pixel = 8  # complex float32
    image_offset = 0
    pixel_offset = bytes_per_pixel
    line_offset = xsize * bytes_per_pixel

    print(f"Writing VRT to {outfile}")
    for band_num, filename in enumerate(file_list, start=1):
        source_filename = os.path.abspath(filename)
        print(f"Adding {source_filename}")
        options = [
            "subClass=VRTRawRasterBand",
            f"SourceFilename={source_filename}",
            f"ImageOffset={image_offset}",
            f"PixelOffset={pixel_offset}",
            f"LineOffset={line_offset}",
        ]
        out_raster.AddBand(gdal_dtype, options)

        # Set the metada on the newly created bands
        date = _get_date(filename)
        metadata = {
            "Date": date,
            "AcquisitionTime": date,
            "Wavelength": str(SENTINEL_WAVELENGTH),
        }
        band = out_raster.GetRasterBand(band_num)
        metadata_domain = "slc"
        band.SetMetadata(metadata, metadata_domain)

    out_raster = None  # Force write


def create_vrt_stack_manual(
    file_list, subset_bbox=None, target_extent=None, outfile="slcs_base.vrt", use_abs_path=True,
):
    # Use the first file in the stack to get size, transform info
    ds = gdal.Open(file_list[0])
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    ds = None
    print("write vrt file for stack directory")
    xoff, yoff, xsize_sub, ysize_sub = get_subset_bbox(
        xsize, ysize, subset_bbox=subset_bbox, target_extent=target_extent
    )
    with open(outfile, "w") as fid:
        fid.write(f'<VRTDataset rasterXSize="{xsize_sub}" rasterYSize="{ysize_sub}">\n')

        for idx, filename in enumerate(file_list, start=1):
            if use_abs_path:
                filename = os.path.abspath(filename)
            date = _get_date(filename)
            outstr = f"""    <VRTRasterBand dataType="CFloat32" band="{idx}">
        <SimpleSource>
            <SourceFilename>{filename}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="CFloat32"/>
            <SrcRect xOff="{xoff}" yOff="{yoff}" xSize="{xsize_sub}" ySize="{ysize_sub}"/>
            <DstRect xOff="0" yOff="0" xSize="{xsize_sub}" ySize="{ysize_sub}"/>
        </SimpleSource>
        <Metadata domain="slc">
            <MDI key="Date">{date}</MDI>
            <MDI key="Wavelength">{SENTINEL_WAVELENGTH}</MDI>
            <MDI key="AcquisitionTime">{date}</MDI>
        </Metadata>
    </VRTRasterBand>\n"""
            fid.write(outstr)

        fid.write("</VRTDataset>")


def get_subset_bbox(xsize, ysize, subset_bbox=None, target_extent=None):
    """Get the subset bounding box for a given target extent.

    Parameters
    ----------
    xsize : int
        size of the x dimension of the image
    ysize : int
        size of the y dimension of the image
    subset_bbox : tuple[int], optional
        Desired bounding box of subset as (left, bottom, right, top)
    target_extent : tuple[int], optional
        Desired bounding box in the `-te` gdal format, (xmin, ymin, xmax, ymax)

    Returns
    -------
    xoff, yoff, xsize_sub, ysize_sub : tuple[int]
    """

    if subset_bbox is not None:
        left, bottom, right, top = subset_bbox
        xoff = left
        yoff = top
        xsize_sub = right - left
        ysize_sub = bottom - top
    elif target_extent is not None:
        # -te xmin ymin xmax ymax
        xoff, yoff, xmax, ymax = target_extent
        xsize_sub = xmax - xoff
        ysize_sub = ymax - yoff
    else:
        xoff, yoff, xsize_sub, ysize_sub = 0, 0, xsize, ysize

    return xoff, yoff, xsize_sub, ysize_sub


def _get_date(filename):
    match = re.search(r"\d{4}\d{2}\d{2}", filename)
    if not match:
        raise ValueError(f"{filename} does not contain date as YYYYMMDD")
    return match.group()


if __name__ == "__main__":
    # Parse command line
    args = get_cli_args()

    # Get ann list and slc list
    if args.in_vrts is not None:
        file_list = sorted(args.in_vrts)
    elif args.in_vrts_file is not None:
        with open(args.in_vrts_file) as f:
            file_list = f.read().splitlines().sort()
    else:
        raise ValueError("Need to pass either --in-vrts or --in-vrts-file")

    num_slc = len(file_list)
    print("Number of SLCs Used: ", num_slc)

    # Set up single stack file
    Path(args.out_dir).mkdir(exist_ok=True, parents=True)
    outfile = os.path.join(args.out_dir, args.out_vrt_name)
    # create_vrt_stack(file_list, outfile=outfile)
    create_vrt_stack_manual(
        file_list,
        outfile=outfile,
        subset_bbox=args.subset_bbox,
        target_extent=args.target_extent,
    )
