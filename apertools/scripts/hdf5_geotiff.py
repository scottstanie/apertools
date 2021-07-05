import subprocess
import apertools.sario
import apertools.utils
import apertools.kml
import apertools.log

logger = apertools.log.get_log()


def _log_and_run(cmd):
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)


def rescale_dset(infile, scale, nodata):
    logger.info(f"Rescaling data by {scale}")
    cmd = (
        f"gdal_calc.py --quiet -A {infile} --outfile=tmp_out.tif"
        f' --calc="A * {scale}" --NoDataValue={nodata}'
    )
    _log_and_run(cmd)
    _log_and_run(f"mv tmp_out.tif {infile}")


def hdf5_to_geotiff(
    infile,
    rsc=None,
    output=None,
    dset=None,
    nodata="0",
    outtype="Float32",
    unit=None,
    scale=None,
    convert_to_cumulative=False,
):

    if rsc:
        rsc_data = apertools.sario.load(rsc)
    else:
        try:
            rsc_data = apertools.sario.load_dem_from_h5(infile)
        except:
            raise ValueError(".rsc loading failed")
            # rsc_data = None

    north, south, east, west = apertools.kml.rsc_nsew(rsc_data)
    ullr_string = "%f %f %f %f" % (west, north, east, south)

    if output is None:
        file_ext = apertools.utils.get_file_ext(infile)
        output = infile.replace(file_ext, ".tif")

    # create_geotiff(rsc_data, kml_file, img_filename, shape='box', outfile)
    # TODO: fix up the kml one...
    cmd = (
        'gdal_translate -sds -a_nodata "{nodata}" -a_srs EPSG:4326 '
        " -a_ullr {ullr} {input} {out} -ot {outtype}"
    )
    if dset is None:
        # tmp_out gets split per band
        cmd1 = cmd.format(
            ullr=ullr_string,
            input=infile,
            out="tmp_out.tif",
            nodata=nodata,
            outtype=outtype,
        )
        _log_and_run(cmd1)
        cmd2 = (
            f"gdal_merge.py -separate -ot {outtype} -o {output} "
            f' -n "{nodata}" -a_nodata "{nodata}" tmp_out*tif'
        )
        _log_and_run(cmd2)
        _log_and_run("rm tmp_out*.tif")
    else:
        instring = """ HDF5:"{}"://{} """.format(infile, dset)
        cmd1 = cmd.format(
            ullr=ullr_string,
            input=instring,
            out=output,
            nodata=nodata,
            outtype=outtype,
        )
        _log_and_run(cmd1)

    if unit is not None:
        logger.info(f"Setting unit to {unit}")
        apertools.sario.set_unit(output, unit)

    if scale is not None and scale != 1:
        rescale_dset(output, scale, nodata)

    if convert_to_cumulative:
        logger.info(f"Converting velocities to cumulative deformation")
        slclist = apertools.sario.load_slclist_from_h5(infile, dset=dset)
        scale = apertools.utils.velo_to_cumulative_scale(slclist)
        rescale_dset(output, scale, nodata)
