import h5py
import os
from apertools import sario

OLD_TO_NEW = {
    "geo_dates": "slc_dates",
    "int_dates": "ifg_dates",
    "geo": "slc",
    "geo_sum": "slc_sum",
    "igram": "ifg",
    "igram_sum": "ifg_sum",
}


def fix_prep(filepath):
    for fname in ["unw_stack.h5", "masks.h5"]:
        f = os.path.join(filepath, fname)
        print("running rename on ", f)
        rename_dsets(f)
        print("running fix on ", f)
        fix_datasets(f)


def rename_dsets(h5file):
    with h5py.File(h5file, "a") as hf:
        for old, new in OLD_TO_NEW.items():
            if old in hf:
                hf[new] = hf[old]
                for k, v in hf[old].attrs.items():
                    hf[new].attrs[k] = v
                del hf[old]


def fix_datasets(h5file):
    # Check which are in the current file
    with h5py.File(h5file) as hf:
        redo_ifgs = "ifg_dates" in hf
        redo_slcs = "slc_dates" in hf
        redo_latlon = "dem_rsc" in hf and "lat" not in hf

    if redo_latlon:
        rsc_data = sario.load_dem_from_h5(h5file)
        sario.save_latlon_to_h5(h5file, rsc_data=rsc_data)
    if redo_ifgs:
        dl = load_ifgstr(h5file, parse=True)
        sario.save_ifglist_to_h5(out_file=h5file, ifg_date_list=dl, overwrite=True)
    if redo_slcs:
        dl = load_slcstr(h5file, parse=True)
        sario.save_slclist_to_h5(out_file=h5file, slc_date_list=dl, overwrite=True)

    latlon_dsets = ["slc", "slc_sum", "ifg", "ifg_sum", "stack_flat_shifted"]
    depth_dims = ["slc_dates", None, "ifg_idx", None, "ifg_idx"]
    dsets = []
    with h5py.File(h5file) as hf:
        for name, dim in zip(latlon_dsets, depth_dims):
            if name in hf:
                dsets.append((name, dim))

    for name, dim in dsets:
        print(f"attaching {dim} in {name} to depth")
        sario.attach_latlon(h5file, name, depth_dim=dim)


def load_slcstr(h5file, dset=None, parse=True):
    with h5py.File(h5file, "r") as f:
        if dset is None:
            slclist_str = f["slc_dates"][()].astype(str)
        else:
            slclist_str = f[dset].attrs["slc_dates"][()].astype(str)

    if parse:
        return sario.parse_slclist_strings(slclist_str)
    else:
        return slclist_str


def load_ifgstr(h5file, dset=None, parse=True):
    with h5py.File(h5file, "r") as f:
        date_pair_strs = f["ifg_dates"][:].astype(str)
    if parse:
        return sario.parse_ifglist_strings(date_pair_strs)
    else:
        return date_pair_strs
