# prepare inputs
# dt_obj = datetime(2020, 12, 25, 14, 7, 44)
rows, cols = inc.shape
lon = ds.lon.data
lat = ds.lat.data
dt_arr = pd.to_datetime(ds.date.data).to_pydatetime()

atr = {
    'LENGTH' : rows,
    'WIDTH'  : cols,
    'X_FIRST': lon[0],
    'Y_FIRST': lat[0],
    'X_STEP' : (lon[1] - lon[0]).item(),
    'Y_STEP' : (lat[1] - lat[0]).item(),
}

def compute_los_tide(dt_obj, atr):
    # compute SET via pysolid
    (tide_e, tide_n, tide_u) = pysolid.calc_solid_earth_tides_grid(
        dt_obj, atr, display=False, verbose=False
    )

    # project SET from ENU to radar line-of-sight (LOS) direction with positive for motion towards satellite
    # inc_angle = 34.0 / 180.0 * np.pi  # radian, typical value for Sentinel-1
    # head_angle = (
    #     -168.0 / 180.0 * np.pi
    # )  # radian, typical value for Sentinel-1 desc track
    tide_los = -1 * (tide_e * los_enu[0] + tide_n * los_enu[1] + tide_u * los_enu[2])
    return 100 * tide_los # convert to cm

from tqdm import tqdm
tides = []
for dt in tqdm(dt_arr):
    # print(f"Computing {dt}")
    tides.append(compute_los_tide(dt, atr))


# TODO: need the actual acquisition time, not just the date
tides_stack = np.stack(tides)
ds_tides = xr.DataArray(
    data=tides_stack,
    dims=("date", "lat", "lon"),
    coords={
        "date": ds.date,
        "lat": ds.lat,
        "lon": ds.lon,
    },
)
ds_tides