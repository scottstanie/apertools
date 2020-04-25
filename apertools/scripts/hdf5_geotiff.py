import subprocess
import apertools.sario
import apertools.utils
import apertools.kml
import apertools.log
logger = apertools.log.get_log()


def hdf5_to_geotiff(
    infile,
    rsc=None,
    output=None,
    dset=None,
    nodata="0",
    outtype="Float32",
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
    cmd = ('gdal_translate -sds -a_nodata "{nodata}" -a_srs EPSG:4326 '
           ' -a_ullr {ullr} {input} {out}')
    if dset is None:
        # tmp_out gets split per band
        cmd1 = cmd.format(
            ullr=ullr_string,
            input=infile,
            out="tmp_out.tif",
            nodata=nodata,
        )
        logger.info("running:")
        logger.info(cmd1)
        subprocess.check_call(cmd1, shell=True)
        cmd2 = (f'gdal_merge.py -separate -ot {outtype} -o {output} '
                f' -n "{nodata}" -a_nodata "{nodata}" tmp_out*tif')
        logger.info("running:")
        logger.info(cmd2)
        subprocess.check_call(cmd2, shell=True)
        subprocess.check_call("rm tmp_out*.tif", shell=True)
    else:
        instring = ''' HDF5:"{}"://{} '''.format(infile, dset)
        cmd1 = cmd.format(
            ullr=ullr_string,
            input=instring,
            out=output,
            nodata=nodata,
        )
        logger.info("running:")
        logger.info(cmd1)
        subprocess.check_call(cmd1, shell=True)