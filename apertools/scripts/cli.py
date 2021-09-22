"""
Main command line entry point to manage all other sub commands
"""
import os
from os.path import abspath, join
import glob
import json
import click
import subprocess
from collections import Counter

# import apertools
from datetime import datetime

import apertools.log

logger = apertools.log.get_log()


def _log_and_run(cmd):
    logger.info(cmd)
    subprocess.check_call(cmd, shell=True)


# Main entry point `aper`:
@click.group()
@click.option("--verbose", is_flag=True)
@click.option(
    "--path",
    type=click.Path(exists=False, file_okay=False, writable=True),
    default=".",
    help="Path of interest for command. "
    "Will search for files path or change directory, "
    "depending on command.",
)
@click.pass_context
def cli(ctx, verbose, path):
    """Command line tools for processing insar."""
    # Store these to be passed to all sub commands
    ctx.obj = {}
    ctx.obj["verbose"] = verbose
    ctx.obj["path"] = path


# COMMAND: view-stack
@cli.command("view-stack")
@click.argument("filename")
@click.option(
    "--dset", default="stack", help="Dataset within hdf5 file", show_default=True
)
@click.option(
    "--cmap",
    default="seismic_wide_y",
    help="Colormap for image display.",
    show_default=True,
)
@click.option(
    "--label",
    default="Centimeters",
    help="Label on colorbar/yaxis for plot",
    show_default=True,
)
@click.option("--title", help="Title for image plot")
@click.option("--row-start", type=int)
@click.option("--row-end", type=int)
@click.option("--col-start", type=int)
@click.option("--col-end", type=int)
@click.option(
    "--rowcol",
    help="Use row,col for legened entries (instead of default lat,lon)",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option("--vmax", type=float)
@click.option("--vmin", type=float)
def view_stack(
    filename,
    dset,
    cmap,
    label,
    title,
    row_start,
    row_end,
    col_start,
    col_end,
    rowcol,
    vmax,
    vmin,
):
    """Explore timeseries on stack of deformation images."""
    import apertools.sario, apertools.latlon, apertools.plotting
    import numpy as np

    try:
        import hdf5plugin
    except ImportError:
        pass
    import h5py

    ext = os.path.splitext(filename)[1]
    if ext == ".h5":
        with h5py.File(filename, "r") as f:
            deformation = f[dset][:]
            slclist = apertools.sario.load_slclist_from_h5(filename, dset)
    elif ext == ".nc":
        import xarray as xr
        import pandas as pd

        ds = xr.open_dataset(filename)
        if dset not in ds.data_vars:
            print(f"WARNING: {dset} not a data variable in {filename}")
            dset = list(ds.data_vars)[0]
            print(f"Using {dset} isntead")
        deformation = ds[dset]
        date_dim = deformation.dims[0]
        slclist = ds[date_dim].values
        # convert from numpy datetime64 to datetime.datetime objects
        slclist = pd.to_datetime(slclist).tz_localize("utc").to_pydatetime()

    if slclist is None or deformation is None:
        return

    if rowcol:
        rsc_data = None
    else:
        if ext == ".h5":
            rsc_data = apertools.sario.load_dem_from_h5(filename)
        elif ext == ".nc":
            lats = ds[deformation.dims[1]].values
            lons = ds[deformation.dims[2]].values
            rsc_data = apertools.latlon.grid_to_rsc(lons, lats, sparse=True)
            # TODO: to in want to try and lazy load?

    # deformation = apertools.latlon.LatlonImage(data=deformation, rsc_data=rsc_data)
    if any((row_start, row_end, col_start, col_end)):
        deformation = deformation[:, row_start:row_end, col_start:col_end]
    img = np.mean(deformation[-3:], axis=0)

    apertools.plotting.view_stack(
        deformation,
        img,
        slclist=slclist,
        title=title,
        label=label,
        cmap=cmap,
        lat_lon=not rowcol,
        rsc_data=rsc_data,
        vmax=vmax,
        vmin=vmin,
    )


# COMMAND: plot
@cli.command("plot")
@click.argument("filename")
@click.option("--downsample", default=1, help="Amount to downsample image")
@click.option("--cmap", default="dismph", help="Colormap for image display.")
@click.option("--band", default=1, show_default=True, help="Band to load (for GDAL)")
@click.option("--title", help="Title for image plot")
@click.option(
    "--colorbar/--no-colorbar", default=True, help="Display colorbar on figure"
)
def plot(filename, downsample, band, cmap, title, colorbar):
    """Quick plot of a single InSAR file.

    filename: Name of InSAR file to plot (possible extensions: .int, .cor, .unw, .geo,...)"

    Can downsample for easier viewing.
    """
    # from .plot_insar import plot_image
    import matplotlib.pyplot as plt
    from apertools.plotting import plot_ifg

    img = apertools.sario.load(filename, downsample=downsample, band=band)
    plot_ifg(img, title=title, colorbar=colorbar)
    plt.show(block=True)


# COMMAND: kml
@cli.command()
@click.argument("imgfile", required=False)
@click.option(
    "--shape",
    default="box",
    help="kml shape: use 'box' for image overlay, 'polygon' for geojson square",
)
@click.option("--rsc", help=".rsc file containing lat/lon start and steps")
@click.option(
    "--geojson", "-g", help="Optional: if making shape from .geojson, file to specify"
)
@click.option("--title", "-t", help="Title of the KML object once loaded.")
@click.option("--desc", "-d", help="Description for google Earth.")
@click.option("--output", "-o", help="File to save kml output to")
@click.option("--cmap", default="seismic", help="Colormap (if saving .npy image)")
@click.option(
    "--normalize", is_flag=True, default=False, help="Center image to [-1, 1]"
)
@click.option("--vmax", type=float, help="Maximum value for imshow")
@click.option("--vmin", type=float, help="Minimum value for imshow")
@click.pass_obj
def kml(
    context,
    imgfile,
    shape,
    rsc,
    geojson,
    title,
    desc,
    output,
    cmap,
    normalize,
    vmax,
    vmin,
):
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
        apertools.sario.save(
            new_filename,
            image,
            cmap=cmap,
            normalize=normalize,
            preview=True,
            vmax=vmax,
            vmin=vmin,
        )

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
@click.option(
    "--pause",
    "-p",
    default=200,
    help="For --animate, time in milliseconds to pause"
    " between stack layers (default 200).",
)
@click.option(
    "--save",
    "-s",
    help="If you want to save the animation as a movie," " title to save file as.",
)
@click.option(
    "--display/--no-display",
    help="Pop up matplotlib figure to view (instead of just saving)",
    default=True,
)
@click.option("--cmap", default="seismic", help="Colormap for image display.")
@click.option(
    "--shifted/--no-shifted", default=True, help="Shift colormap to be 0 centered."
)
@click.option(
    "--file-ext", help="If not loading deformation.npy, the extension of files to load"
)
@click.option(
    "--ifglist/--no-ifglist",
    default=False,
    help="If loading other file type, also load `ifglist` file  for titles",
)
@click.option(
    "--db/--no-db", help="Use dB scale for images (default false)", default=False
)
@click.option("--vmax", type=float, help="Maximum value for imshow")
@click.option("--vmin", type=float, help="Minimum value for imshow")
@click.pass_obj
def animate(
    context, pause, save, display, cmap, shifted, file_ext, ifglist, db, vmin, vmax
):
    """Creates animation for 3D image stack.

    If deformation.npy and slclist.npy or .unw files are not in current directory,
    use the --path option:

        aper --path /path/to/igrams animate

    Note: Default is to load 3D stack named deformation.npy
    Otherwise, use --file-ext "unw", for example, to grab all files
    """
    import apertools.plotting
    import apertools.sario

    if file_ext:
        stack = apertools.sario.load_stack(directory=context["path"], file_ext=file_ext)
        titles = sorted(glob(os.path.join(context["path"], "*" + file_ext)))
    else:
        slclist, deformation = apertools.sario.load_deformation(context["path"])
        stack = deformation
        titles = [d.strftime("%Y-%m-%d") for d in slclist]

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
@cli.command("dem-rate")
@click.option("--rsc-file", help="name of .rsc file")
@click.option("--orig-rsc-file", help="name of original bigger (pre-looked) .rsc file")
def dem_rate(rsc_file, orig_rsc_file):
    """Print the upsample rate of a dem

    aper dem-rate   # Looks in current folder for one .rsc file
    aper dem-rate /path/to/dem.rsc
    aper dem-rate /path/to/dem.rsc /path/to/big_elevation.dem.rsc  # prints num looks

    """
    import apertools.sario
    import apertools.utils

    # full_file = join(context['path'], rsc_file)
    if rsc_file is None:
        rsc_file = apertools.sario.find_rsc_file(directory=".")

    x_uprate, y_uprate = apertools.sario.calc_upsample_rate(rsc_filename=rsc_file)

    click.echo(
        "%s has (%.5f, %.5f) times the default spacing in (x, y)"
        % (rsc_file, x_uprate, y_uprate)
    )

    default_spacing = 30.0
    click.echo(
        "This is equal to (%.2f, %.2f) meter spacing between pixels"
        % (default_spacing / x_uprate, default_spacing / y_uprate)
    )
    if orig_rsc_file is not None:
        orig_x_uprate, orig_y_uprate = apertools.sario.calc_upsample_rate(
            rsc_filename=orig_rsc_file
        )
        click.echo(
            "(%.0f, %.0f) looks were taken on %s to get %s"
            % (
                orig_x_uprate / x_uprate,
                orig_y_uprate / y_uprate,
                orig_rsc_file,
                rsc_file,
            )
        )


# COMMAND: overlaps
@cli.command()
@click.option(
    "--sentinel-path", default=".", help="Path to directory containing .SAFE folders"
)
@click.option(
    "--filename",
    "-f",
    help="(full) path to the dem.rsc or .geojson file. If not "
    "included, will output all files matching the path/date criteria",
)
@click.option(
    "--path-num", "-p", type=int, help="Select one orbit path number for overlaps"
)
@click.option(
    "--end-date", "-e", help="Cut off Sentinel files after this date (format: YYYYMMDD)"
)
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

    sent_files = glob.glob(join(sentinel_path, "*.SAFE"))
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

    logger.info(
        "%d Sentinel .SAFE files overlap within specified date range" % len(sent_list)
    )

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
@click.option(
    "--bands",
    "-b",
    type=int,
    multiple=True,
    default=[1],
    help="Specify which bands within file to include in VRT",
)
@click.option(
    "--interleave",
    type=click.Choice(["BIP", "BIL", "BSQ"]),
    default=None,
    help="Type of pixel interleave in binary file",
)
@click.option("--num-bands", type=int, help="Number of bands in file")
@click.option(
    "--out-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=".",
    help="Location of output file",
    show_default=True,
)
@click.option("--metadata", "-m", multiple=True)
@click.option("--metadata-domain")
def save_vrt(
    filenames,
    rsc_file,
    cols,
    rows,
    dtype,
    bands,
    interleave,
    num_bands,
    out_dir,
    metadata,
    metadata_domain,
):
    """Save GDAL .vrt file for easier binary raster loading

    List as many filenames with the same rsc as necessary
    """
    import apertools.sario

    if metadata:
        print("metadata dict:", _args_to_dict(metadata))
    for f in filenames:
        outfile = os.path.join(out_dir, os.path.split(f)[1]) + ".vrt"
        apertools.sario.save_vrt(
            filename=f,
            rows=rows,
            cols=cols,
            dtype=dtype,
            rsc_file=rsc_file,
            bands=bands,
            interleave=interleave,
            num_bands=num_bands,
            outfile=outfile,
            relative=False,
            metadata_dict=_args_to_dict(metadata),
            metadata_domain=metadata_domain,
        )


def _args_to_dict(args):
    if len(args) % 2 != 0:
        raise ValueError("Must pass pairse of key/value pairs")
    return dict(zip(args[::2], args[1::2]))


@cli.command("smallslc")
@click.argument("filenames", nargs=-1)
@click.option("--rsc-file", help="If exists, the .rsc file of data")
@click.option(
    "--downrate",
    default=10,
    help="Downsampling beyond the 30m SRTM resolution",
    show_default=True,
)
@click.option(
    "--resample",
    default="nearest",
    type=click.Choice(
        ["nearest", "bilinear", "cubic", "cubicspline", "lanczos", "average", "mode"]
    ),
    help="GDAL Resampling method",
    show_default=True,
)
@click.option("--out-dir", "-o", type=click.Path(exists=True), default="./")
@click.option("--overwrite/--no-overwrite", default=False)
def smallslc(
    filenames,
    rsc_file,
    downrate,
    resample,
    out_dir,
    overwrite,
):
    """Save complex binary .geo/.slc as small version for quick look in dismph

    List as many filenames with the same rsc as necessary
    """
    import apertools.sario

    x_uprate, y_uprate = apertools.sario.calc_upsample_rate(rsc_filename=rsc_file)
    pct_x = int(100 / x_uprate / downrate)
    pct_y = int(100 / y_uprate / downrate)
    for f in filenames:
        dest_name = os.path.split(f)[1]
        # For ROI_PAC driver, it only recognizes .slc, not .geo
        dest_path = os.path.join(out_dir, "small_" + dest_name.replace(".geo", ".slc"))
        if os.path.exists(dest_path) and not overwrite:
            print(f"{dest_path} exists, overwrite={overwrite}: skipping.")
            continue

        apertools.sario.save_vrt(
            filename=f,
            rsc_file=rsc_file,
        )
        if resample == "average":
            col_looks = round(downrate * x_uprate)
            row_looks = round(downrate * y_uprate)
            print(f"Downlooking {f} by {row_looks}, {col_looks} into {dest_path}")
            apertools.sario.save(
                dest_path,
                apertools.sario.load(
                    f,
                    looks=(row_looks, col_looks),
                    separate_complex=True,
                ),
            )
            apertools.sario.save(
                dest_path + ".rsc",
                apertools.sario.load(rsc_file, looks=(row_looks, col_looks)),
            )
        else:
            cmd = (
                f"gdal_translate -of ROI_PAC {f + '.vrt'} {dest_path}"
                f" -r {resample} -outsize {pct_x}% {pct_y}% "
            )
            _log_and_run(cmd)


@cli.command("looked-dem")
@click.option(
    "--src-dem",
    default="../elevation.dem",
    help="Original, large DEM",
    show_default=True,
)
@click.option(
    "--dest-rsc",
    default="dem.rsc",
    help=".rsc file of the destination",
    show_default=True,
)
@click.option(
    "--outname", default="elevation_looked.dem", help="destination", show_default=True
)
def looked_dem(src_dem, dest_rsc, outname):
    """Save a smaller DEM version to match size of dest-rsc file"""
    import apertools.sario as sario

    rsc = sario.load(dest_rsc)
    xstep, ystep = rsc["x_step"], rsc["y_step"]
    # -r nearest == Use nearest-neighbor resampling, -tr = target resolution
    cmd = (
        f"gdal_translate -r nearest -of ROI_PAC -tr {xstep} {ystep} {src_dem} {outname}"
    )
    _log_and_run(cmd)


@cli.command("hdf5-gtiff")
@click.argument("infile")
@click.option("--output", "-o", help="output geotiff file")
@click.option("--rsc", help=".rsc file containing lat/lon start and steps")
@click.option("--dset", help="specify to only save one dataset")
@click.option("--nodata", default="0", show_default=True)
@click.option("--outtype", default="Float32", show_default=True)
@click.option("--unit", default=None, show_default=True)
@click.option(
    "--scale",
    default=1.0,
    show_default=True,
    help="Multiply pixels by this number to calculate new values in output file",
)
@click.option("--convert-to-cumulative", is_flag=True)
def geotiff(
    infile, rsc, output, dset, nodata, outtype, unit, scale, convert_to_cumulative
):
    from .hdf5_geotiff import hdf5_to_geotiff

    return hdf5_to_geotiff(
        infile,
        rsc,
        output,
        dset,
        nodata,
        outtype,
        unit,
        scale,
        convert_to_cumulative,
    )


@cli.command("set-unit")
@click.argument("filenames", nargs=-1)
@click.option("--unit", "-u", default="cm", help="unit for file", show_default=True)
@click.option("--band", "-b", help="Specify only 1 band to set unit")
def set_unit(filenames, unit, band):
    """Alter the metadata of gdal-readable file to add units"""
    import apertools.sario

    for f in filenames:
        apertools.sario.set_unit(f, unit=unit, band=band)


@cli.command("az-inc-to-enu")
@click.argument("infile")
@click.option("--outfile", "-o", default="los_enu.tif", show_default=True)
def convert_to_enu(infile, outfile):
    import apertools.utils

    apertools.utils.az_inc_to_enu(infile, outfile)


@cli.command("mask-by-elevation")
@click.argument("filenames", nargs=-1)
@click.option("--dem", "-d", help="DEM filename")
@click.option("--cutoff", type=float, help="Elevation threshold at which to mask")
@click.option(
    "--operator",
    type=click.Choice([">", "<"]),
    default=">",
    help="operator to use for masking "
    "(e.g. using '>' will mask all values where the dem is greater "
    "than the cutoff, keeping only the small values",
    show_default=True,
)
@click.option(
    "--largest-component",
    is_flag=True,
    help="Keep only the largest component which survives cutoff",
)
def mask_by_elevation(filenames, dem, cutoff, operator, largest_component):
    """Set NoData pixels in files based on elevation threshold"""
    import apertools.utils
    import rasterio as rio

    for f in filenames:
        cmd = (
            f"gdal_calc.py --quiet -A {f} -B {dem} --outfile=tmp_out.tif "
            f' --calc="A * ~(B {operator} {cutoff})" --NoDataValue=0'
        )
        _log_and_run(cmd)
        _log_and_run(f"mv tmp_out.tif {f}")
        # _log_and_run(f"gdal_edit.py -a_nodata 0 {f}")

        if largest_component is True:
            logger.info("Keeping only pixels from largest connected component")
            with rio.open(f) as src:
                img = src.read(1)
            binimg = img != 0
            fg_idxs = apertools.utils.find_largest_component_idxs(binimg, strel_size=3)
            mask_fname = "idxs.bin"
            with rio.open(
                mask_fname,
                "w",
                count=1,
                driver="ENVI",
                height=fg_idxs.shape[0],
                width=fg_idxs.shape[1],
                dtype="uint8",
            ) as dst:
                dst.write(fg_idxs.astype("uint8"), 1)

            cmd = (
                f"gdal_calc.py --quiet -A {f} -B {mask_fname} --outfile=tmp_out.tif "
                f' --calc="A * B " --NoDataValue=0'
            )
            _log_and_run(cmd)
            _log_and_run(f"mv tmp_out.tif {f}")
            _log_and_run(f"rm {mask_fname} {mask_fname.replace('.bin', '.hdr')}")


@cli.command("subset")
@click.option(
    "--bbox", nargs=4, type=float, help="Window lat/lon bounds: left bot right top"
)
@click.option("--out-dir", "-o", type=click.Path(exists=True))
@click.option("--in-dir", "-i", type=click.Path(exists=True))
@click.option("--start-date", "-s", type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end-date", "-e", type=click.DateTime(formats=["%Y-%m-%d"]))
def subset(bbox, out_dir, in_dir, start_date, end_date):
    """Read window subset from .geos in another directory

    Writes the smaller .geos to `outpath`, along with the
    extra files going with it (elevation.dem, .orbtimings)
    """
    import apertools.sario
    from apertools.utils import force_symlink
    import apertools.subset

    if abspath(out_dir) == abspath(in_dir):
        raise ValueError("--in-dir cannot be same as --out-dir")

    # dems:
    apertools.subset.copy_subset(
        join(in_dir, "elevation.dem"),
        join(out_dir, "elevation.dem"),
        bbox=bbox,
        driver="ROI_PAC",
    )
    # Fortran cant read anything but 15-space .rsc file :|
    apertools.sario.save("elevation.dem.rsc", apertools.sario.load("elevation.dem.rsc"))

    # weird params file
    with open(join(out_dir, "params"), "w") as f:
        f.write(f"{join(abspath(out_dir), 'elevation.dem')}\n")
        f.write(f"{join(abspath(out_dir), 'elevation.dem.rsc')}\n")

    # geos and .orbtimings
    for in_fname in glob.glob(join(in_dir, "*.geo.vrt")):
        cur_date = apertools.sario.parse_slclist_strings(os.path.split(in_fname)[1])
        if (end_date is not None and cur_date > end_date.date()) or (
            start_date is not None and cur_date < start_date.date()
        ):
            continue
        img = apertools.subset.read_subset(bbox, in_fname, driver="VRT")

        _, nameonly = os.path.split(in_fname)
        out_fname = join(out_dir, nameonly).replace(".vrt", "")
        # Can't write vrt?
        # copy_subset(bbox, in_fname, out_fname, driver="VRT")
        click.echo(f"Subsetting {in_fname} to {out_fname}")
        apertools.sario.save(out_fname, img)

        src, dest = (
            in_fname.replace(".geo.vrt", ".orbtiming"),
            out_fname.replace(".geo", ".orbtiming"),
        )

        click.echo(f"symlinking {src} to {dest}")
        force_symlink(src, dest)
        # copyfile(s, d)


@cli.command("subset-vrt")
@click.argument("filenames", nargs=-1)
@click.option(
    "--bbox", nargs=4, type=float, help="Window lat/lon bounds: left bot right top"
)
@click.option(
    "--out-dir", "-o", type=click.Path(exists=True), default=".", show_default=True
)
# @click.option("--start-date", "-s", type=click.DateTime(formats=["%Y-%m-%d"]))
# @click.option("--end-date", "-e", type=click.DateTime(formats=["%Y-%m-%d"]))
def subset_vrt(filenames, bbox, out_dir):  # , start_date, end_date):
    import apertools.subset

    for f in filenames:
        out_fname = os.path.join(out_dir, os.path.split(f)[1] + ".vrt")
        apertools.subset.copy_vrt(f, out_fname=out_fname, bbox=bbox, verbose=True)


from apertools.netcdf import run_hdf5_to_netcdf

cli.add_command(run_hdf5_to_netcdf)


@cli.command("shift-pixel")
@click.argument("in_filename")
@click.argument("out_filename")
@click.option(
    "--full-pixel",
    default=False,
    help="Shift by a full pixel, instead of half",
    is_flag=True,
)
@click.option(
    "--down-right",
    default=False,
    help="Shift image down and right, instead of up/left",
    is_flag=True,
)
def shift_by_pixel(in_filename, out_filename, full_pixel, down_right):
    """Shift an image by a full (or half) pixel up and to the left (or down/right)"""
    import apertools.sario

    apertools.sario.shift_by_pixel(
        in_filename, out_filename, full_pixel=full_pixel, down_right=down_right
    )


@cli.command("nc-to-tif")
@click.argument("in_filename")
@click.argument("out_filename")
@click.option("--dset", help="dataset within the netcdf to write out.")
@click.option(
    "--crs",
    default="epsg:4326",
    help="add a coordinate reference system, if not discovered",
    show_default=True,
)
def nc_to_tif(in_filename, out_filename, dset, crs):
    import xarray as xr

    # import rioxarray
    # xds = rioxarray.open_rasterio
    xds = xr.open_dataset(in_filename)
    da = xds[dset] if dset is not None else xds
    da.rio.write_crs(crs, inplace=True)
    if "lat" in xds:
        da.rio.set_spatial_dims("lon", "lat", inplace=True)
    da.rio.to_raster(out_filename)
