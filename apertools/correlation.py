import numpy as np
import xarray as xr
import apertools.sario as sario
import apertools.utils as utils


# TODO: lat, lon...
def create_cor_matrix(row, col, filename="cor_stack.nc"):
    with xr.open_dataset(filename) as ds:
        filenames = ds.filenames
        corr_values = ds["stack"][:, row, col]

    intlist = np.array(sario.parse_intlist_strings(filenames))
    geolist = np.array(utils.geolist_from_igrams(intlist))
    full_igram_list = np.array(utils.full_igram_list(geolist))

    valid_idxs = np.searchsorted(
        sario.intlist_to_filenames(full_igram_list),
        sario.intlist_to_filenames(intlist),
    )

    full_corr_values = np.nan * np.ones((len(full_igram_list),))
    full_corr_values[valid_idxs] = corr_values

    # Now arange into a matrix: fill the upper triangle
    ngeos = len(geolist)
    out = np.full((ngeos, ngeos), np.nan)

    # Diagonal is always 1
    rdiag, cdiag = np.diag_indices(ngeos)
    out[rdiag, cdiag] = 1

    rows, cols = np.triu_indices(ngeos, k=1)
    out[rows, cols] = full_corr_values
    return out


# TODO: plot with dates
