import numpy as np
import os
import apertools.sario as sario
import xarray as xr

if __name__ == "__main__":
    fname = "slc_stack.nc"
    if os.path.exists(fname):
        data = xr.open_dataarray("slc_stack.nc")
    else:
        data = sario.save_slc_amp_stack(directory=".", ext=".slc", outname=fname)

    seasonal = data.groupby("date.season").mean()
    seasonal.plot(
        x="lon",
        y="lat",
        col="season",
        col_wrap=2,
        vmax=np.percentile(seasonal.data, 95),
        cmap="gray",
        #     cmap="discrete_seismic7",
    )