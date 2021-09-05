import os
import time
import datetime
import numpy as np
import h5py
import netCDF4 as nc
import rasterio as rio
import click

from apertools import latlon, sario
from apertools.log import get_log, log_runtime

logger = get_log()


# TODO: do i ever need to do a list? just make 2 files, right?
@log_runtime
def hdf5_to_netcdf(
    filename,
    dset_name="stack",
    stack_dim="idx",
    outname=None,
    data_units=None,
    copy_data_attrs=True,
    bbox=None,
):
    """Convert the stack in HDF5 to NetCDF with appropriate metadata"""

    if not filename.endswith(".h5"):
        raise ValueError(f"{filename} must be an HDF5 file")

    if outname is None:
        outname = filename.replace(".h5", ".nc")
    if not outname.endswith(".nc"):
        raise ValueError(f"{outname} must be an .nc filename")

    with h5py.File(filename) as hf:
        # Get data and references from HDF5 file

        # Just get one example for shape
        if dset_name not in hf:
            raise ValueError(
                f"Requested dset: {dset_name}. "
                f"Dset keys available in {filename}: {list(hf.keys())}"
            )
        dset = hf[dset_name]
        nstack = dset.shape[0] if dset.ndim == 3 else 1
        rows, cols = dset.shape[-2:]
        print(nstack, rows, cols)
        if "lat" in hf and "lon" in hf:
            lat_arr = hf["lat"][()]
            lon_arr = hf["lon"][()]
        else:  # Try using the rsc_data
            lon_arr, lat_arr = latlon.get_latlon_arrs(h5_filename=filename, bbox=bbox)

        (row_top, row_bot), (col_left, col_right) = latlon.window_rowcol(
            lon_arr, lat_arr, bbox=bbox
        )
        lat_arr = lat_arr[row_top:row_bot]
        lon_arr = lon_arr[col_left:col_right]
        rows, cols = len(lat_arr), len(lon_arr)

        if stack_dim in ("date", "slc_dates"):
            try:
                slclist = sario.load_slclist_from_h5(filename, dset=dset_name)
            except KeyError:  # Give this one a shot too
                slclist = sario.load_slclist_from_h5(filename)
            stack_dim_arr = to_datetimes(slclist)
        else:
            stack_dim_arr = np.arange(nstack)

        # TODO: store the int dates as dims... somehow

        logger.info("Making dimensions and variables")
        mode = "a" if os.path.exists(outname) else "w"
        with nc.Dataset(outname, mode) as f:
            # f.set_auto_mask(False)
            # f.set_always_mask(False)
            if dset_name in f.variables:
                logger.info("%s already in %s: skipping.", dset_name, outname)
                return

            f.history = "Updated " + time.ctime(time.time())
            if "lat" in f.variables:
                latitudes = f["lat"]
            else:
                f.createDimension("lat", rows)
                latitudes = f.createVariable("lat", "f4", ("lat",), zlib=True)
                latitudes.units = "degrees north"
                latitudes[:] = lat_arr
            if "lon" in f.variables:
                longitudes = f["lon"]
            else:
                f.createDimension("lon", cols)
                longitudes = f.createVariable("lon", "f4", ("lon",), zlib=True)
                longitudes.units = "degrees east"
                longitudes[:] = lon_arr

            if nstack > 1 and stack_dim not in f.variables:
                f.createDimension(stack_dim, nstack)
                if stack_dim == "date":
                    idxs = f.createVariable(stack_dim, "f4", (stack_dim,), zlib=True)
                    idxs.units = f"days since {slclist[0]}"
                else:
                    idxs = f.variables[stack_dim]

                if stack_dim == "date":
                    d2n = nc.date2num(stack_dim_arr, units=idxs.units)
                    idxs[:] = d2n
                else:
                    idxs[:] = stack_dim_arr

            # Finally, the actual stack
            # stackvar = rootgrp.createVariable("stack/1", "f4", ("date", "lat", "lon"))
            logger.info(f"Writing data to {outname}:{dset_name}")
            if hf[dset_name].dtype == np.dtype("bool"):
                bool_type = "i1"
                # bool_type = f.createEnumType(
                #     np.uint8, "bool_t", {"FALSE": 0, "TRUE": 1}
                # )
                dt = bool_type
                # fill_value = 0
            else:
                dt = hf[dset_name].dtype
                # fill_value = 0
            # Note on fill: seems like when i set fill to 0, it masks it..
            # and turns 0s into nans

            data_var = f.createVariable(
                dset_name,
                dt,
                (stack_dim, "lat", "lon") if nstack > 1 else ("lat", "lon"),
                # fill_value=fill_value,
                zlib=True,
            )
            if data_units:
                data_var.units = data_units
            if copy_data_attrs:
                for k, v in dset.attrs.items():
                    # Manual skip for the slclist or ifglist, or dimension scale stuff
                    if "dates" in k or 'dimension' in k.lower():
                        continue
                    setattr(data_var, k, v)
            if nstack > 1:
                d = dset[:, row_top:row_bot, col_left:col_right]
            else:
                d = dset[row_top:row_bot, col_left:col_right]
            logger.info(f"d shape: {d.shape}")
            data_var[:] = d


def to_datetimes(date_list):
    return [datetime.datetime(*d.timetuple()[:6]) for d in date_list]


def create_empty_nc_stack(
    outname,
    stack_dim_name="idx",
    stack_data_name="stack",
    depth=None,
    date_list=None,
    file_list=None,
    gdal_file=None,
    dem_rsc_file=None,
    bbox=None,
    dtype="float32",
    lat_units="degrees north",
    lon_units="degrees east",
    overwrite=False,
):
    """Creates skeleton of .nc stack without writing stack data

    See create_nc_stack for details
    """
    if not outname.endswith(".nc"):
        raise ValueError(f"{outname} must be an .nc filename")

    lon_arr, lat_arr = latlon.get_latlon_arrs(
        dem_rsc_file=dem_rsc_file, gdal_file=gdal_file, bbox=bbox
    )

    (row_top, row_bot), (col_left, col_right) = latlon.window_rowcol(
        lon_arr, lat_arr, bbox=bbox
    )
    lat_arr = lat_arr[row_top:row_bot]
    lon_arr = lon_arr[col_left:col_right]
    rows, cols = len(lat_arr), len(lon_arr)

    # Get data and references from HDF% file

    # TODO: unlimited....
    # TODO: store filename metadata?
    if stack_dim_name == "date":
        if date_list is None:
            raise ValueError("Need 'date_list' if 3rd dimension is 'date'")
        stack_dim_arr = to_datetimes(date_list)
    else:
        if depth is None:
            if file_list is not None:
                depth = len(file_list)
        stack_dim_arr = np.arange(depth)

    logger.info("Making dimensions and variables")
    open_mode = "w" if overwrite else "a"

    with nc.Dataset(outname, open_mode) as f:
        f.history = "Created " + time.ctime(time.time())
        if file_list is not None:
            f.setncattr_string("filenames", file_list)

        f.createDimension("lat", rows)
        f.createDimension("lon", cols)
        # Could make this unlimited to add to it later?
        latitudes = f.createVariable("lat", "f4", ("lat",), zlib=True)
        longitudes = f.createVariable("lon", "f4", ("lon",), zlib=True)
        latitudes.units = lat_units
        longitudes.units = lon_units

        #######
        # for dset_name, stack_dim_name, stack_arr in zip(
        # dset = hf[dset_name]
        # nstack, _, _ = dset.shape

        # TODO: will i ever add like this? and need to check?
        # if stack_dim_name not in f.dimensions:
        f.createDimension(stack_dim_name, depth)
        # if stack_dim_name not in f.variables:
        if stack_dim_name == "date":
            stack_dim_variable = f.createVariable(
                stack_dim_name, "f4", (stack_dim_name,), zlib=True
            )
            stack_dim_variable.units = f"days since {date_list[0]}"
        else:
            stack_dim_variable = f.createVariable(
                stack_dim_name, "i4", (stack_dim_name,)
            )

        # Write data
        latitudes[:] = lat_arr
        longitudes[:] = lon_arr
        if stack_dim_name == "date":
            d2n = nc.date2num(stack_dim_arr, units=stack_dim_variable.units)
            stack_dim_variable[:] = d2n
        else:
            stack_dim_variable[:] = stack_dim_arr

        # Finally, the actual stack
        # stackvar = rootgrp.createVariable("stack/1", "f4", ("date", "lat", "lon"))
        logger.info(f"Writing dummy data for {stack_data_name}")
        if np.dtype(dtype) == np.dtype("bool"):
            bool_type = "i1"
            # bool_type = f.createEnumType(
            #     np.uint8, "bool_t", {"FALSE": 0, "TRUE": 1}
            # )
            dt = bool_type
            fill_value = 0
        else:
            dt = np.dtype(dtype)
            fill_value = 0

        f.createVariable(
            stack_data_name,
            dt,
            (stack_dim_name, "lat", "lon"),
            fill_value=fill_value,
            zlib=True,
        )
        # d = dset[:, row_top:row_bot, col_left:col_right]
        # logger.info(f"d shape: {d.shape}")
        # stackvar[:] = d


@log_runtime
def create_nc_stack(
    outname,
    file_list,
    band=2,
    stack_dim_name="idx",
    stack_data_name="stack",
    depth=None,
    date_list=None,
    gdal_file=None,
    dem_rsc_file=None,
    bbox=None,
    dtype="float32",
    lat_units="degrees north",
    lon_units="degrees east",
    overwrite=False,
    use_gdal=False,
    gdal_driver=None,
):
    """Create a NetCDF file with lat/lon dimensions written, but no data yet

    Prepares for stack creation with a data file specified later

    Args:
        outname (str): name of .nc output file to save
        file_list (list[str]): if the layers come from files, the list of files
        band (int): the gdal band number to read from file_list
        stack_dim_name (str): default = "idx". Name of the 3rd dimension of the stack
            (Dimensions are (stack_dim_name, lat, lon) )
            If stack_dim_name="date", Need "date_list" passed.
        stack_data_name (str): default="stack", name of the data variable in the file
        depth (int): number of layers which will appear in output stack
            if date_list passed, will use len(date_list)
            if file_list passed, will use len(file_list)
            if None and no date_list, will create an unlimited dimension
        date_list (list[datetime.date]): if layers of stack correspond to dates,
            the list of dates to use for the coordinates
        gdal_file (str): filname with same lat/lon grid as desired new .nc file
        dem_rsc_file (str): .rsc file containing information for the desired output lat/lon grid
        bbox (tuple[float]): bounding box if using a subset of the lat/lons provided
        dtype: default="float32", the numpy datatype of the stack data
        lat_units (str): default = "degrees north",
        lon_units (str): default = "degrees east",
        overwrite (bool): default = False, will overwrite file if true
        use_gdal (bool): default = False, use gdal to read each file of `file_list`
        gdal_driver (str): optional, driver for opening files

    """
    file_list = sorted(file_list)
    create_empty_nc_stack(
        outname,
        stack_dim_name=stack_dim_name,
        stack_data_name=stack_data_name,
        depth=depth,
        date_list=date_list,
        file_list=file_list,
        gdal_file=gdal_file,
        dem_rsc_file=dem_rsc_file,
        bbox=bbox,
        dtype=dtype,
        lat_units=lat_units,
        lon_units=lon_units,
        overwrite=overwrite,
    )

    with nc.Dataset(outname, "a") as f:
        stack_var = f.variables[stack_data_name]
        chunk_depth, chunk_rows, chunk_cols = stack_var.chunking()
        depth, rows, cols = stack_var.shape

        buf = np.empty((chunk_depth, rows, cols), dtype=dtype)
        lastidx = 0
        cur_chunk_size = 0  # runs from 0 to chunk_depth
        for idx, in_fname in enumerate(file_list):
            if idx % 100 == 0:
                logger.info(f"Processing {in_fname} -> {idx+1} out of {len(file_list)}")

            if idx % chunk_depth == 0 and idx > 0:
                logger.info(f"Writing {lastidx}:{lastidx+chunk_depth}")
                stack_var[lastidx : lastidx + cur_chunk_size, :, :] = buf

                cur_chunk_size = 0
                lastidx = idx

            if use_gdal:
                with rio.open(in_fname, driver=gdal_driver) as fin:
                    # now store this in the buffer until emptied
                    data = fin.read(band)
            else:
                data = sario.load(in_fname)

            curidx = idx % chunk_depth
            cur_chunk_size += 1
            buf[curidx, :, :] = data

        if cur_chunk_size > 0:
            # Write the final part of the buffer:
            stack_var[lastidx : lastidx + cur_chunk_size, :, :] = buf[:cur_chunk_size]


# Create the command-line version
@click.command("hdf5-to-netcdf")
@click.argument("filename")
@click.option("--outname", "-o", type=click.Path(dir_okay=False))
@click.option(
    "--stack-dset",
    help="Name of 3d datasets in the .h5 file to convert",
)
@click.option(
    "--stack-dim",
    help="Name of 3rd dimension for stack. If 'date' passed, will try to load date list",
)
@click.option(
    "--bbox",
    nargs=4,
    type=float,
    help="To only convert a subset, pass lat/lon bounds: left bot right top",
)
def run_hdf5_to_netcdf(filename, outname, stack_dset, stack_dim, bbox):
    hdf5_to_netcdf(filename, stack_dset, stack_dim, outname, bbox)
