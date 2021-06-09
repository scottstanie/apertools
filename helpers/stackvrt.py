#!/usr/bin/env python
import re
import argparse
import os
from osgeo import gdal
import apertools.utils as utils

SENTINEL_WAVELENGTH = 0.05546576


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


def create_vrt_stack_manual(file_list, outfile="slcs_base.vrt"):
    # Use the first file in the stack to get size, transform info
    ds = gdal.Open(file_list[0])
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    ds = None
    print("write vrt file for stack directory")
    with open(outfile, "w") as fid:
        fid.write(f'<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">\n')

        for idx, filename in enumerate(file_list, start=1):
            date = _get_date(filename)
            outstr = f"""    <VRTRasterBand dataType="CFloat32" band="{idx}">
        <SimpleSource>
            <SourceFilename>{filename}</SourceFilename>
            <SourceBand>1</SourceBand>
            <SourceProperties RasterXSize="{xsize}" RasterYSize="{ysize}" DataType="CFloat32"/>
            <SrcRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/>
            <DstRect xOff="0" yOff="0" xSize="{xsize}" ySize="{ysize}"/>
        </SimpleSource>
        <Metadata domain="slc">
            <MDI key="Date">{date}</MDI>
            <MDI key="Wavelength">{SENTINEL_WAVELENGTH}</MDI>
            <MDI key="AcquisitionTime">{date}</MDI>
        </Metadata>
    </VRTRasterBand>\n"""
            fid.write(outstr)

        fid.write("</VRTDataset>")


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
    utils.mkdir_p(args.out_dir)
    outfile = os.path.join(args.out_dir, args.out_vrt_name)
    # create_vrt_stack(file_list, outfile=outfile)
    create_vrt_stack_manual(file_list, outfile=outfile)
