"""
Main command line entry point to manage all other sub commands
"""
import os
import glob
import json
import click
import subprocess
from collections import Counter
# import apertools
import h5py
from datetime import datetime

import apertools.log
logger = apertools.log.get_log()


# Main entry point `aper`:
@click.group()
@click.option('--verbose', is_flag=True)
@click.option('--path',
              type=click.Path(exists=False, file_okay=False, writable=True),
              default='.',
              help="Path of interest for command. "
              "Will search for files path or change directory, "
              "depending on command.")
@click.pass_context
def cli(ctx, verbose, path):
    """Command line tools for processing insar."""
    # Store these to be passed to all sub commands
    ctx.obj = {}
    ctx.obj['verbose'] = verbose
    ctx.obj['path'] = path


# COMMAND: view-stack
@cli.command('view-stack')
@click.option("--filename", default='deformation.npy', help="Name of saved deformation stack")
@click.option("--cmap", default='seismic', help="Colormap for image display.")
@click.option("--label", default='Centimeters', help="Label on colorbar/yaxis for plot")
@click.option("--title", help="Title for image plot")
@click.option('--row-start', default=0)
@click.option('--row-end', default=-1)
@click.option('--col-start', default=0)
@click.option('--col-end', default=-1)
@click.option("--rowcol",
              help="Use row,col for legened entries (instead of default lat,lon)",
              is_flag=True,
              default=False)
@click.pass_obj
def view_stack(context, filename, cmap, label, title, row_start, row_end, col_start, col_end,
               rowcol):
    """Explore timeseries on deformation image.

    If deformation.npy and geolist.npy or .unw files are not in current directory,
    use the --path option:

        aper --path /path/to/igrams view_stack

    """
    import numpy as np
    if filename.endswith(".h5"):
        h_file = h5py.File(filename, "r")
        deformation = h_file["deformation"]
        geolist = [datetime.strptime(g.decode("ascii"), "%Y%m%d").date() for g in h_file["geolist"]]
    else:
        geolist, deformation = apertools.sario.load_deformation(context['path'], filename=filename)

    if geolist is None or deformation is None:
        return

    if rowcol:
        rsc_data = None
    else:
        rsc_data = apertools.sario.load(os.path.join(context['path'], 'dem.rsc'))

    deformation = apertools.latlon.LatlonImage(data=deformation, dem_rsc=rsc_data)
    deformation = deformation[:, row_start:row_end, col_start:col_end]
    img = np.mean(deformation[-3:], axis=0)

    apertools.plotting.view_stack(
        deformation,
        img,
        geolist=geolist,
        title=title,
        label=label,
        cmap=cmap,
        lat_lon=not rowcol,
    )


# COMMAND: plot
@cli.command('plot')
@click.argument("filename")
@click.option("--downsample", default=1, help="Amount to downsample image")
@click.option("--cmap", default='dismph', help="Colormap for image display.")
@click.option("--title", help="Title for image plot")
@click.option("--alpha",
              default=0.6,
              help="Transparency for background magnitude (if plotting insar)")
@click.option("--colorbar/--no-colorbar", default=True, help="Display colorbar on figure")
def plot(filename, downsample, cmap, title, alpha, colorbar):
    """Quick plot of a single InSAR file.

    filename: Name of InSAR file to plot (possible extensions: .int, .cor, .unw, .geo,...)"

    Can downsample for easier viewing.
    Requires a .dem.rsc file to also be in same directory

    If not using the current directory, use the --path option:

        aper --path /path/to/igrams <filename>

    """
    from .plot_insar import plot_image
    img = apertools.sario.load(filename, downsample=downsample)
    plot_image(img, title=title, colorbar=colorbar, alpha=alpha)


# COMMAND: kml
@cli.command()
@click.argument("imgfile", required=False)
@click.option("--shape",
              default="box",
              help="kml shape: use 'box' for image overlay, 'polygon' for geojson square")
@click.option("--rsc", help=".rsc file containing lat/lon start and steps")
@click.option("--geojson", "-g", help="Optional: if making shape from .geojson, file to specify")
@click.option("--title", "-t", help="Title of the KML object once loaded.")
@click.option("--desc", "-d", help="Description for google Earth.")
@click.option("--output", "-o", help="File to save kml output to")
@click.option("--cmap", default="seismic", help="Colormap (if saving .npy image)")
@click.option("--normalize", is_flag=True, default=False, help="Center image to [-1, 1]")
@click.option("--vmax", type=float, help="Maximum value for imshow")
@click.option("--vmin", type=float, help="Minimum value for imshow")
@click.pass_obj
def kml(context, imgfile, shape, rsc, geojson, title, desc, output, cmap, normalize, vmax, vmin):
    """Creates .kml file for some image
    IMGFILE is the image to load into Google Earth

    Example:

        aper kml 20180420_20180502.tif --rsc dem.rsc -t "My igram" -d "Kiluea eruption" -o out.kml

    """
    def _save_npy_file(imgfile, new_filename):
        image = apertools.sario.load(imgfile)
        if image.ndim > 2:
            # For 3D stack, assume we just want the final image
            image = image[-1]
            logger.info("Saving final image of stack")
        apertools.sario.save(new_filename,
                             image,
                             cmap=cmap,
                             normalize=normalize,
                             preview=True,
                             vmax=vmax,
                             vmin=vmin)

    if geojson:
        with open(geojson) as f:
            gj_dict = json.load(f)
    else:
        gj_dict = None

    rsc_data = apertools.sario.load(rsc) if rsc else None
    # Check if imgfile is a .npy saved matrix
    file_ext = apertools.utils.get_file_ext(imgfile)
    if file_ext in (".npy", ".h5"):
        new_filename = imgfile.replace(file_ext, ".png")
        _save_npy_file(imgfile, new_filename)
    else:
        new_filename = imgfile

    kml_string = apertools.kml.create_kml(
        rsc_data=rsc_data,
        img_filename=new_filename,
        gj_dict=gj_dict,
        title=title,
        desc=desc,
        kml_out=output,
        shape=shape,
    )
    print(kml_string)


# COMMAND: animate
@cli.command()
@click.option("--pause",
              '-p',
              default=200,
              help="For --animate, time in milliseconds to pause"
              " between stack layers (default 200).")
@click.option("--save",
              '-s',
              help="If you want to save the animation as a movie,"
              " title to save file as.")
@click.option("--display/--no-display",
              help="Pop up matplotlib figure to view (instead of just saving)",
              default=True)
@click.option("--cmap", default='seismic', help="Colormap for image display.")
@click.option("--shifted/--no-shifted", default=True, help="Shift colormap to be 0 centered.")
@click.option("--file-ext", help="If not loading deformation.npy, the extension of files to load")
@click.option("--intlist/--no-intlist",
              default=False,
              help="If loading other file type, also load `intlist` file  for titles")
@click.option("--db/--no-db", help="Use dB scale for images (default false)", default=False)
@click.option("--vmax", type=float, help="Maximum value for imshow")
@click.option("--vmin", type=float, help="Minimum value for imshow")
@click.pass_obj
def animate(context, pause, save, display, cmap, shifted, file_ext, intlist, db, vmin, vmax):
    """Creates animation for 3D image stack.

    If deformation.npy and geolist.npy or .unw files are not in current directory,
    use the --path option:

        aper --path /path/to/igrams animate

    Note: Default is to load 3D stack named deformation.npy
    Otherwise, use --file-ext "unw", for example, to grab all files
    """
    import apertools.plotting
    import apertools.sario
    if file_ext:
        stack = apertools.sario.load_stack(directory=context['path'], file_ext=file_ext)
        titles = sorted(apertools.sario.find_files(context['path'], "*" + file_ext))
    else:
        geolist, deformation = apertools.sario.load_deformation(context['path'])
        stack = deformation
        titles = [d.strftime("%Y-%m-%d") for d in geolist]

    if db:
        stack = apertools.utils.db(stack)

    apertools.plotting.animate_stack(
        stack,
        pause_time=pause,
        display=display,
        titles=titles,
        save_title=save,
        cmap_name=cmap,
        shifted=shifted,
        vmin=vmin,
        vmax=vmax,
    )


# COMMAND: dem-rate
@cli.command('dem-rate')
@click.option("--rsc-file", help="name of .rsc file")
def dem_rate(rsc_file):
    """Print the upsample rate of a dem

        aper dem-rate   # Looks in current folder for one .rsc file
        aper dem-rate /path/to/dem.rsc

    """
    import apertools.sario
    import apertools.utils
    # full_file = os.path.join(context['path'], rsc_file)
    if rsc_file is None:
        rsc_file = apertools.sario.find_rsc_file(directory=".")

    x_uprate, y_uprate = apertools.utils.calc_upsample_rate(rsc_filename=rsc_file)

    click.echo("%s has (%.5f, %.5f) times the default spacing in (x, y)" %
               (rsc_file, x_uprate, y_uprate))

    default_spacing = 30.0
    click.echo("This is equal to (%.2f, %.2f) meter spacing between pixels" %
               (default_spacing / x_uprate, default_spacing / y_uprate))


# COMMAND: overlaps
@cli.command()
@click.option("--sentinel-path", default=".", help="Path to directory containing .SAFE folders")
@click.option("--filename",
              "-f",
              help="(full) path to the dem.rsc or .geojson file. If not "
              "included, will output all files matching the path/date criteria")
@click.option("--path-num", "-p", type=int, help="Select one orbit path number for overlaps")
@click.option("--end-date", "-e", help="Cut off Sentinel files after this date (format: YYYYMMDD)")
@click.option("--start-date", "-s", help="Cut off Sentinel files before this date")
def overlaps(sentinel_path, filename, path_num, start_date, end_date):
    """List all Sentinel .SAFEs overlapping with area

    --filename can either look at a DEM using .rsc file, or the bounding
    box of a .geojson file

    Note that the --sentinel-path must contain .SAFE folders, not .zip,
    since the map overlay kml file must be extracted

    Logging information sent to stderr, find overlap files printed to stdout
    To save, just redirect stdout to a file:

        aper overlaps --filename box.geojson > overlap_files.txt
    """
    import apertools.parsers
    import apertools.sario

    def _parse(date_string):
        return datetime.strptime(date_string, "%Y%m%d").date()

    def _log_paths(sent_list):
        path_counter = Counter([s.path for s in sent_list])
        logger.info("Number of files per path:")
        for path, num in path_counter.items():
            logger.info("Path %d: %d files" % (path, num))

    logger.info("Searching %s for Sentinel .SAFE files" % sentinel_path)
    if not os.path.exists(sentinel_path):
        raise ValueError("%s does not exist" % sentinel_path)

    sent_files = glob.glob(os.path.join(sentinel_path, "*.SAFE"))
    sent_list = [apertools.parsers.Sentinel(s) for s in sent_files]
    logger.info("%d Sentinel .SAFE files found" % len(sent_list))

    _log_paths(sent_list)

    if path_num:
        logger.info("Selecting path %s only" % path_num)
        sent_list = [s for s in sent_list if s.path == path_num]

    if start_date:
        logger.info("Filtering out files before %s" % start_date)
        sent_list = [s for s in sent_list if s.date >= _parse(start_date)]
    if end_date:
        logger.info("Filtering out files after %s" % end_date)
        sent_list = [s for s in sent_list if s.date <= _parse(end_date)]

    logger.info("%d Sentinel .SAFE files overlap within specified date range" % len(sent_list))

    if filename:
        logger.info("Searching %s for .rsc or .geojson file" % filename)
        area = apertools.sario.load(filename)
        sent_list = [s for s in sent_list if s.overlaps(area)]
        logger.info("%d Sentinel .SAFE files overlap with area" % len(sent_list))

    logger.info("Final path count:")
    _log_paths(sent_list)

    print("\n".join(s.filename for s in sent_list))


# COMMAND: save-vrt
@cli.command("save-vrt")
@click.argument("filenames", nargs=-1)
@click.option("--rsc-file", help="If exists, the .rsc file of data")
@click.option("--cols", type=int, help="Number of columns (width) in file")
@click.option("--rows", type=int, help="Number of rows (file_length) in file")
@click.option("--dtype", help="Optional number dtype string")
@click.option("--band", type=int, help="The band number to use for the VRT")
@click.option("--num-bands", type=int, help="Number of bands in file")
def save_vrt(filenames, rsc_file, cols, rows, dtype, band, num_bands):
    """Save GDAL .vrt file binary raster loading

    List as many filenames with the same rsc as necessary
    """
    import apertools.sario
    for f in filenames:
        apertools.sario.save_as_vrt(
            filename=f,
            rows=rows,
            cols=cols,
            dtype=dtype,
            rsc_file=rsc_file,
            band=band,
            num_bands=num_bands,
        )


# COMMAND: save-vrt
@cli.command("smallslc")
@click.argument("filenames", nargs=-1)
@click.option("--rsc-file", help="If exists, the .rsc file of data")
@click.option("--downrate", default=10, help="Downsampling beyond the 30m SRTM resolution")
def smallslc(
    filenames,
    rsc_file,
    downrate,
):
    """Save complex binary .geo/.slc as small version for quick look in dismph

    List as many filenames with the same rsc as necessary
    """
    import apertools.sario
    import apertools.utils
    x_uprate, y_uprate = apertools.utils.calc_upsample_rate(rsc_filename=rsc_file)
    pct_x = int(100 / x_uprate / downrate)
    pct_y = int(100 / y_uprate / downrate)
    for f in filenames:
        dest = "small_" + f
        # For ROI_PAC driver, it only recognizes .slc, not .geo
        dest = dest.replace(".geo", ".slc")
        apertools.sario.save_as_vrt(
            filename=f,
            rsc_file=rsc_file,
        )
        cmd = """gdal_translate -of ROI_PAC {src} {dest} -outsize {px}% {py}% """.format(
            src=f + ".vrt",
            dest=dest,
            px=pct_x,
            py=pct_y,
        )
        logger.info(cmd)
        subprocess.check_call(cmd, shell=True)


@cli.command("hdf5-gtiff")
@click.argument("infile")
@click.option("--output", "-o", help="output geotiff file")
@click.option("--rsc", help=".rsc file containing lat/lon start and steps")
@click.option("--dset", help="specify to only save one dataset")
@click.option("--nodata", default="0")
@click.option("--outtype", default="Float32")
def geotiff(infile, rsc, output, dset, nodata, outtype):
    import apertools.sario
    import apertools.utils
    import apertools.kml

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
    cmd = 'gdal_translate -sds -a_nodata "{nodata}" -a_srs EPSG:4326 -a_ullr {ullr} {input} {out}'
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
        cmd2 = 'gdal_merge.py -separate -ot {ot} -o {output} -n "{nd}" -a_nodata "{nd}" tmp_out*tif'.format(
            nd=nodata,
            ot=outtype,
            output=output,
        )
        logger.info("running:")
        logger.info(cmd2)
        subprocess.check_call(cmd2, shell=True)
        subprocess.check_call("rm tmp_out*.tif", shell=True)
    else:
        cmd1 = cmd.format(
            ullr=ullr_string,
            input=infile,
            out=output,
            nodata=nodata,
        )
        logger.info("running:")
        logger.info(cmd1)
        subprocess.check_call(cmd1, shell=True)
