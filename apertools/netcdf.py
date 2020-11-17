import time
import datetime
import numpy as np
import h5py

from apertools import latlon, sario


def hdf5_to_netcdf(
    filename,
    stack_dset_list=["stack"],
    stack_dim_list=["idx"],
    outname=None,
    bbox=None,
):
    """Convert the stack in HDF5 to NetCDF with appropriate metadata"""
    import netCDF4 as nc

    if not filename.endswith(".h5"):
        raise ValueError(f"{filename} must be an HDF5 file")

    if outname is None:
        outname = filename.replace(".h5", ".nc")
    if not outname.endswith(".nc"):
        raise ValueError(f"{outname} must be an .nc filename")

    with h5py.File(filename) as hf:
        # Get data and references from HDF% file

        # Just get one example for shape
        if any(d not in hf for d in stack_dset_list):
            raise ValueError(
                f"Requested dsets: {stack_dset_list}. Dset keys available in {filename}: {list(hf.keys())}"
            )
        nstack, rows, cols = hf[stack_dset_list[0]].shape
        lon_arr, lat_arr = latlon.get_latlon_arrs(h5_filename=filename, bbox=bbox)

        (row_top, row_bot), (col_left, col_right) = latlon.window_rowcol(
            lon_arr, lat_arr, bbox=bbox
        )
        lat_arr = lat_arr[row_top:row_bot]
        lon_arr = lon_arr[col_left:col_right]
        rows, cols = len(lat_arr), len(lon_arr)

        stack_arrs = []
        for dset_name, stack_dim in zip(stack_dset_list, stack_dim_list):
            nstack, _, _ = hf[dset_name].shape
            if stack_dim == "date":
                try:
                    geolist = sario.load_geolist_from_h5(filename, dset=dset_name)
                except KeyError:  # Give this one a shot too
                    geolist = sario.load_geolist_from_h5(filename)
                stack_dim_arr = to_datetimes(geolist)
            else:
                stack_dim_arr = np.arange(nstack)
            stack_arrs.append(stack_dim_arr)

        # TODO: store the int dates as dims... somehow

        print("Making dimensions and variables")
        with nc.Dataset(outname, "w") as f:
            f.history = "Created " + time.ctime(time.time())

            f.createDimension("lat", rows)
            f.createDimension("lon", cols)
            # Could make this unlimited to add to it later?
            latitudes = f.createVariable("lat", "f4", ("lat",), zlib=True)
            longitudes = f.createVariable("lon", "f4", ("lon",), zlib=True)
            latitudes.units = "degrees north"
            longitudes.units = "degrees east"

            for dset_name, stack_dim_name, stack_arr in zip(
                stack_dset_list, stack_dim_list, stack_arrs
            ):
                dset = hf[dset_name]
                nstack, _, _ = dset.shape
                if stack_dim_name not in f.dimensions:
                    f.createDimension(stack_dim_name, nstack)
                if stack_dim_name not in f.variables:
                    if stack_dim_name == "date":
                        idxs = f.createVariable(
                            stack_dim_name, "f4", (stack_dim_name,), zlib=True
                        )
                        idxs.units = f"days since {geolist[0]}"
                    else:
                        idxs = f.createVariable(stack_dim_name, "i4", (stack_dim_name,))

                # Write data
                latitudes[:] = lat_arr
                longitudes[:] = lon_arr
                if stack_dim_name == "date":
                    d2n = nc.date2num(stack_arr, units=idxs.units)
                    idxs[:] = d2n
                else:
                    idxs[:] = stack_arr

                # Finally, the actual stack
                # stackvar = rootgrp.createVariable("stack/1", "f4", ("date", "lat", "lon"))
                print(f"Writing {dset_name} data")
                if hf[dset_name].dtype == np.dtype("bool"):
                    bool_type = "i1"
                    # bool_type = f.createEnumType(
                    #     np.uint8, "bool_t", {"FALSE": 0, "TRUE": 1}
                    # )
                    dt = bool_type
                    fill_value = 0
                else:
                    dt = hf[dset_name].dtype
                    fill_value = 0

                stackvar = f.createVariable(
                    dset_name,
                    dt,
                    (stack_dim_name, "lat", "lon"),
                    fill_value=fill_value,
                    zlib=True,
                )
                d = dset[:, row_top:row_bot, col_left:col_right]
                print(f"d shape: {d.shape}")
                stackvar[:] = d


# TODO: put elsewhere
def to_datetimes(date_list):
    return [datetime.datetime(*d.timetuple()[:6]) for d in date_list]
