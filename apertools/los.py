"""Functions to deal with Line of sight vector computation
"""
from __future__ import division, print_function
import os
import glob
import numpy as np
from numpy import sin, cos
import h5py
import subprocess
# from scipy import interpolate
import sardem.loading

from apertools import latlon, sario

from apertools.log import get_log
import matplotlib.pyplot as plt
logger = get_log()


def find_enu_coeffs(lon, lat, geo_path=None, los_map_file=None, verbose=False):
    """For arbitrary lat/lon, find the coefficients for ENU components of LOS vector

    Args:
        lon (float): longitude of point to get LOS vector
        lat (float): latitude of point
        geo_path (str): path to the directory with the sentinel
            timeseries inversion (contains line-of-sight deformation.npy, dem.rsc,
            and has .db files one directory higher)

    Returns:
        ndarray: enu_coeffs, shape = (3,) array [alpha_e, alpha_n, alpha_up]
        Pointing from ground to satellite
        Can be used to project an ENU vector into the line of sight direction
    """

    if los_map_file is not None:
        return _read_los_map_file(los_map_file, lat, lon)

    los_file = os.path.realpath(os.path.join(geo_path, 'los_vector_%s_%s.txt' % (lon, lat)))
    if not os.path.exists(los_file):
        db_path = _find_db_path(geo_path)
        record_xyz_los_vector(lon,
                              lat,
                              db_path=db_path,
                              outfile=los_file,
                              clear=True,
                              verbose=verbose)

    enu_coeffs = los_to_enu(los_file)

    # Note: vectors are from sat to ground, so uplift is negative
    return -1 * enu_coeffs[0]


def _read_los_map_file(los_map_file, lat, lon):
    with h5py.File(los_map_file) as f:
        lats = f["lats"]
        lons = f["lons"]
        row = _find_nearest(lats, lat)
        col = _find_nearest(lons, lon)
        return f["stack"][:, row, col]


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _find_db_path(geo_path):
    extra_path = os.path.join(geo_path, 'extra_files')
    if os.path.exists(extra_path) and len(glob.glob(extra_path + "/*.db*")) > 0:
        return extra_path
    else:
        return geo_path


def find_east_up_coeffs(geo_path):
    """Find the coefficients for east and up components for LOS vector at midpoint of scene

    Args:
        geo_path (str): path to the directory with the sentinel
            timeseries inversion (contains line-of-sight deformation.npy, dem.rsc,
            and has .db files one directory higher)

    Returns:
        ndarray: east_up_coeffs, a 1x2 array [[east_def, up_def]]
        Combined with another path, used for solving east-up deformation.:
            [east_asc,  up_asc;
             east_desc, up_desc]
        Used as the "A" matrix for solving Ax = b, where x is [east_def; up_def]
    """
    # TODO: make something to adjust 'params' file in case we moved it
    geo_path = os.path.realpath(geo_path)
    # Are we doing this in the .geo folder, or the igram folder?
    # rsc_data = sardem.loading.load_dem_rsc(os.path.join(geo_path, 'dem.rsc'), lower=True)
    rsc_data = sardem.loading.load_dem_rsc(os.path.join(geo_path, 'elevation.dem.rsc'), lower=True)

    midpoint = latlon.grid_midpoint(**rsc_data)
    # The path to each orbit's .db files assumed in same directory as elevation.dem.rsc

    los_file = os.path.realpath(os.path.join(geo_path, 'los_vectors.txt'))
    db_path = _find_db_path(geo_path)

    max_corner_difference, enu_coeffs = check_corner_differences(rsc_data, db_path, los_file)
    logger.info(
        "Max difference in ENU LOS vectors for area corners: {:2f}".format(max_corner_difference))
    if max_corner_difference > 0.05:
        logger.warning("Area is not small, actual LOS vector differs over area.")
        logger.info('Corner ENU coeffs:')
        logger.info(enu_coeffs)
    logger.info("Using midpoint of area for line of sight vectors")

    print("Finding LOS vector for midpoint", midpoint)
    record_xyz_los_vector(*midpoint, db_path=db_path, outfile=los_file, clear=True)

    enu_coeffs = los_to_enu(los_file)

    # Get only East and Up out of ENU
    east_up_coeffs = enu_coeffs[:, ::2]
    # -1 multiplied since vectors are from sat to ground, so vert is negative
    return -1 * east_up_coeffs


def record_xyz_los_vector(lon,
                          lat,
                          db_path=".",
                          outfile="./los_vectors.txt",
                          clear=False,
                          verbose=False):
    """Given one (lon, lat) point, find the LOS from Sat to ground

    Function will run through all possible .db files until non-zero vector is computed

    Records answer in outfile, can be read by read_los_output

    Returns:
        db_files [list]: names of files used to find the xyz vectors
        May be multiple if first failed and produced 0s
    """
    if clear:
        open(outfile, 'w').close()
    if verbose:
        logger.info("Recording xyz los vectors for %s, %s" % (lat, lon))

    exec_path = os.path.expanduser("~/sentinel/look_angle/losvec/losvec_yj")
    stationname = "'{} {}'".format(lat, lon)  # Record where vec is from

    cur_dir = os.getcwd()  # To return to after
    db_path = os.path.realpath(db_path)
    # Make db_files an iterator so we can run through them
    db_files = glob.iglob(os.path.join(db_path, "*.db*"))
    try:
        db_file = next(db_files)
    except StopIteration:
        raise ValueError("Bad db_path, no .db files found: {}".format(db_path))

    os.chdir(db_path)
    db_files_used = []
    while True:
        db_files_used.append(db_file)
        # print("Changing to directory {}".format(db_path))
        # usage: losvel_yj file_db lat lon stationname outfilename
        cmd = "{} {} {} {} {} {}".format(exec_path, db_file, lat, lon, stationname, outfile)
        # print("Running command:")
        # print(cmd)
        if verbose:
            logger.info("Checking db file: %s" % db_file)
        subprocess.check_output(cmd, shell=True)
        _, xyz_list = read_los_output(outfile)
        # if not all((any(vector) for vector in xyz_list)):  # Some corner produced 0s
        if not any(xyz_list[-1]):  # corner produced only 0s
            try:
                db_file = next(db_files)
            except StopIteration:
                logger.warning('Ran out of db files!')
                break
        else:
            break

    # print("Returning to {}".format(cur_dir))
    os.chdir(cur_dir)
    return db_files_used


def read_los_output(los_file, dedupe=True):
    """Reads file of x,y,z positions, parses for lat/lon and vectors

    Example line:
     19.0  -155.0
        0.94451263868681301      -0.30776088245682498      -0.11480032487005554
         6399       4259

    Where first line is "gps station position", or "lat lon",
    next line are the 3 LOS vector coordinates from ground to satellite in XYZ,
    next is x (col) position, y (row) position within the DEM grid

    Args:
        los_file (str): Name of file with line of sight vectors
        dedupe (bool): Remove duplicate lat,lon points which may appear
            recorded with 0s from a wrong .db file

    Returns:
        lat_lon_list (list[tuple]): (lat, lon) tuples of points in file
        xyz_list (list[tuple]): (x, y, z) components of line of sight
    """
    def _line_to_floats(line, split_char=None):
        return tuple(map(float, line.split(split_char)))

    with open(los_file) as f:
        los_lines = f.read().splitlines()

    lat_lon_list = [_line_to_floats(line) for line in los_lines[::3]]
    xyz_list = [_line_to_floats(line) for line in los_lines[1::3]]
    return remove_dupes(lat_lon_list, xyz_list) if dedupe else (lat_lon_list, xyz_list)


def remove_dupes(lat_lon_list, xyz_list):
    """De-duplicates list of lat/lons with LOS vectors

    Example:
    >>> ll_list = [(1, 2), (1, 2), (1, 2), (3, 4)]
    >>> xyz_list = [(0,0,0), (1,2,3), (0,0,0), (4, 5, 6)]
    >>> ll2, xyz2 = remove_dupes(ll_list, xyz_list)
    >>> ll2
    [(1, 2), (3, 4)]
    >>> xyz2
    [(1, 2, 3), (4, 5, 6)]
    """

    latlons, xyzs = [], []
    idx = -1  # Will increment to 1 upon first append
    for lat_lon, xyz in zip(lat_lon_list, xyz_list):
        if lat_lon in latlons:
            # If we added a (0,0,0) vector, check to update it
            if any(xyz) and not any(xyzs[idx]):
                xyzs[idx] = xyz
        else:
            latlons.append(lat_lon)
            xyzs.append(xyz)
            idx += 1
    return latlons, xyzs


def los_to_enu(los_file=None, lat_lons=None, xyz_los_vecs=None):
    """Converts Line of sight vectors from xyz to ENU

    Can read in the LOS vec file, or take a list `xyz_los_vecs`
    Args:
        los_file (str): file to the recorded LOS vector at lat,lon points
        lat_lons (list[tuple[float]]): list of (lat, lon) coordinares for LOS vecs
        xyz_los_vecs (list[tuple[float]]): list of xyz LOS vectors

    Notes:
        Second two args are the result of read_los_output, mutually
        exclusive with los_file

    Returns:
        ndarray: k x 3 ENU 3-vectors
    """
    if los_file:
        lat_lons, xyz_los_vecs = read_los_output(los_file)
    return convert_xyz_latlon_to_enu(lat_lons, xyz_los_vecs)


def convert_xyz_latlon_to_enu(lat_lons, xyz_array):
    return np.array(
        [rotate_xyz_to_enu(xyz, lat, lon) for (lat, lon), xyz in zip(lat_lons, xyz_array)])


def rotate_xyz_to_enu(xyz, lat, lon):
    """Rotates a vector in XYZ coords to ENU

    Args:
        xyz (list[float], ndarray[float]): length 3 x,y,z coordinates, either
            as list of 3, or a 3xk array of k ENU vectors
        lat (float): latitude (deg) of point to rotate into
        lon (float): longitude (deg) of point to rotate into

    Reference: https://gssc.esa.int/navipedia/index.php/\
Transformations_between_ECEF_and_ENU_coordinates

    """
    # Rotate about axis 3 with longitude, then axis 1 with latitude
    R3 = rot(90 + lon, 3, in_degrees=True)
    R1 = rot(90 - lat, 1, in_degrees=True)
    return np.matmul(R1, np.matmul(R3, xyz))


def rot(angle, axis, in_degrees=True):
    """
    Find a 3x3 euler rotation matrix given an angle and axis.

    Rotation matrix used for rotating a vector about a single axis.

    Args:
        angle (float): angle in degrees to rotate
        axis (int): 1, 2 or 3
        in_degrees (bool): specify the angle in degrees. if false, using
            radians for `angle`
    """
    R = np.eye(3)
    if in_degrees:
        angle = np.deg2rad(angle)
    cang = cos(angle)
    sang = sin(angle)
    if axis == 1:
        R[1, 1] = cang
        R[2, 2] = cang
        R[1, 2] = sang
        R[2, 1] = -sang
    elif axis == 2:
        R[0, 0] = cang
        R[2, 2] = cang
        R[0, 2] = -sang
        R[2, 0] = sang
    elif axis == 3:
        R[0, 0] = cang
        R[1, 1] = cang
        R[1, 0] = -sang
        R[0, 1] = sang
    else:
        raise ValueError("axis must be 1, 2 or 2")
    return R


def rotate_enu_to_xyz(enu, lat, lon):
    """Given a vector in ENU, rotate to XYZ coords

    Args:
        enu (list[float], ndarray[float]): E,N,U coordinates, either
            as list of 3, or a 3xk array of k ENU vectors
        lat (float): latitude (deg) of point to rotate into
        lon (float): longitude (deg) of point to rotate into

    Reference: https://gssc.esa.int/navipedia/index.php/\
Transformations_between_ECEF_and_ENU_coordinates


    test_xyz = [-0.127341338217677e7, -0.529776534925940e7,  0.330588991387726e7]
    test_enu = [-1489.929802,   3477268.994159 ,  769.948314]


    """
    # Rotate about axis 3 with longitude, then axis 1 with latitude
    R1 = rot(-(90 - lat), 1, in_degrees=True)
    R3 = rot(-(90 + lon), 3, in_degrees=True)
    R = np.matmul(R1, R3)
    return np.matmul(R, enu)


def project_enu_to_los(enu, los_vec=None, lat=None, lon=None, enu_coeffs=None):
    """Find magnitude of an ENU vector in the LOS direction

    Rotates the line of sight vector to ENU coordinates at
    (lat, lon), then dots with the enu data vector

    Args:
        enu (list[float], ndarray[float]): E,N,U coordinates, either
            as list of 3, or a (3, k) array of k ENU vectors
        los_vec (ndarray[float]) length 3 line of sight, in XYZ frame
        lat (float): degrees latitude of los point
        lon (float): degrees longitude of los point
        enu_coeffs (ndarray) size 3 array of the E,N,U coefficients
        of a line of sight vector. Comes from `find_enu_coeffs`.
            If this arg is used, others are not needed

    Returns:
        ndarray: magnitudes same length as enu input, (k, 1)

    Examples:
    # >>> print('%.2f' % project_enu_to_los([1,2,3],[1, 0, 0], 0, 0))
    # -2.00
    # >>> print('%.2f' % project_enu_to_los([1,2,3],[0, 1, 0], 0, 0))
    # -3.00
    # >>> print('%.2f' % project_enu_to_los([1,2,3],[0, 0, 1], 0, 0))
    # 1.00
    """
    if enu_coeffs is None:
        los_hat = los_vec / np.linalg.norm(los_vec)
        enu_coeffs = rotate_xyz_to_enu(los_hat, lat, lon)
    # import pdb
    # pdb.set_trace()
    return np.dot(enu_coeffs, enu)
    # xyz_vectors = rotate_enu_to_xyz(enu, lat, lon)
    # los_hat = los_vec / np.linalg.norm(los_vec)
    # return np.dot(xyz_vectors.T, los_hat)


def corner_los_vectors(rsc_data, db_path, los_output_file):
    grid_corners = latlon.grid_corners(**rsc_data)
    # clear the output file:
    open(los_output_file, 'w').close()
    db_files_used = []
    for p in grid_corners:
        db_files = record_xyz_los_vector(*p, db_path=db_path, outfile=los_output_file)
        db_files_used.append(db_files)

    return db_files_used, read_los_output(los_output_file)


def check_corner_differences(rsc_data, db_path, los_file):
    """Finds value range for ENU coefficients of the LOS vectors in dem.rsc area

    Used to see if east, north, and up components vary too much for a single value
    to be used to solve for east + vertical part from LOS components
    """
    grid_corners = latlon.grid_corners(**rsc_data)
    # clear the output file:
    open(los_file, 'w').close()
    for p in grid_corners:
        record_xyz_los_vector(*p, db_path=db_path, outfile=los_file)

    enu_coeffs = los_to_enu(los_file)

    # Find range of data for E, N and U
    enu_ranges = np.ptp(enu_coeffs, axis=0)  # ptp = 'peak to peak' aka range
    return np.max(enu_ranges), enu_coeffs


def find_vertical_def(asc_path, desc_path):
    """Calculates vertical deformation for all points in the LOS files

    Args:
        asc_path (str): path to the directory with the ascending sentinel files
            Should contain elevation.dem.rsc, .db files, and igram folder
        desc_path (str): same as asc_path but for descending orbit
    Returns:
        tuple[ndarray, ndarray]: def_east, def_vertical, the two matrices of
            deformation separated by verticl and eastward motion
    """
    asc_path = os.path.realpath(asc_path)
    desc_path = os.path.realpath(desc_path)

    eu_asc = find_east_up_coeffs(asc_path)
    eu_desc = find_east_up_coeffs(desc_path)

    east_up_coeffs = np.vstack((eu_asc, eu_desc))
    print("East-up asc and desc:")
    print(east_up_coeffs)

    asc_igram_path = os.path.join(asc_path, 'igrams')
    desc_igram_path = os.path.join(desc_path, 'igrams')

    asc_geolist, asc_deform = sario.load_deformation(asc_igram_path)
    desc_geolist, desc_deform = sario.load_deformation(desc_igram_path)

    print(asc_igram_path, asc_deform.shape)
    print(desc_igram_path, desc_deform.shape)
    assert asc_deform.shape == desc_deform.shape, 'Asc and desc def images not same size'
    nlayers, nrows, ncols = asc_deform.shape
    # Stack and solve for the East and Up deformation
    d_asc_desc = np.vstack([asc_deform.reshape(-1), desc_deform.reshape(-1)])
    dd = np.linalg.solve(east_up_coeffs, d_asc_desc)
    def_east = dd[0, :].reshape((nlayers, nrows, ncols))
    def_vertical = dd[1, :].reshape((nlayers, nrows, ncols))
    return def_east, def_vertical


def merge_geolists(geolist1, geolist2):
    """Task asc and desc geolists, makes one merged

    Gives the overlap indices of the merged list for each smaller

    """
    merged_geolist = np.concatenate((geolist1, geolist2))
    merged_geolist.sort()

    _, indices1, _ = np.intersect1d(merged_geolist, geolist1, return_indices=True)
    _, indices2, _ = np.intersect1d(merged_geolist, geolist2, return_indices=True)
    return merged_geolist, indices1, indices2


def create_los_map(geo_path, downsample=100, verbose=False, savename="los_enu_map"):
    """Find LOS coeffs at every point in a LatlonImage

    Args:
        geo_path (str): path to the directory with the sentinel
            timeseries inversion (contains line-of-sight deformation.npy, dem.rsc,
            and has .db files one directory higher)
        downsample (int): amount to downsample the .geo dem rsc grid to find points
        verbose (bool): print each point during calculation
        savename (str): base name to save the .npy matrix and a _dem.rsc of the grid data

    Returns:
        ndarray: east_north_up coeffs, a (3, m, n) array
    """
    # TODO: make something to adjust 'params' file in case we moved it
    geo_path = os.path.realpath(geo_path)

    # should we doing this in the .geo folder, or the igram folder?
    rsc_data = sario.load(os.path.join(geo_path, 'elevation.dem.rsc'))
    nrows, ncols = rsc_data['file_length'], rsc_data['width']
    nrows_small = nrows // downsample
    ncols_small = ncols // downsample
    down_rsc = latlon.LatlonImage.crop_rsc_data(
        rsc_data,
        row_start=None,
        col_start=None,
        nrows=nrows_small,
        ncols=ncols_small,
        row_step=downsample,
        col_step=downsample,
    )

    up_img = np.empty((3, nrows_small, ncols_small))
    lons, lats = latlon.grid(sparse=True, **down_rsc)
    lons = lons.reshape(-1)
    lats = lats.reshape(-1)
    logger.info("Finding LOS vector for %s points total" % (len(lats) * len(lons)))
    for rowidx, lat in enumerate(lats.reshape(-1)):
        for colidx, lon in enumerate(lons.reshape(-1)):
            up_img[:, rowidx, colidx] = find_enu_coeffs(lon,
                                                        lat,
                                                        geo_path=geo_path,
                                                        verbose=verbose)

    enu_ll = latlon.LatlonImage(data=up_img, dem_rsc=down_rsc)
    if savename:
        logger.info("Saving data to %s" % savename)
        np.save(savename, enu_ll)
        dem_rsc_file = savename + "_dem.rsc"
        logger.info("Saving dem_rsc to %s" % dem_rsc_file)
        with open(dem_rsc_file, "w") as f:
            f.write(sardem.loading.format_dem_rsc(enu_ll.dem_rsc))

    return enu_ll


def plot_enu_maps(enu_ll, title=None, cmap='jet'):
    fig, axes = plt.subplots(1, 3)

    titles = ['east', 'north', 'up']
    for idx in range(3):
        axim = axes[idx].imshow(np.abs(enu_ll[idx]), vmin=0, vmax=1, cmap=cmap)
        axes[idx].set_title(titles[idx])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axim, cax=cbar_ax)
    fig.suptitle(title)


# def interpolate_coeffs(rsc_data, nrows, ncols, east_up):
#     # This will be if we want to solve the exact coefficients
#     # Make grid to interpolate one
#     grid_corners = latlon.grid_corners(**rsc_data)
#     xx, yy = latlon.grid(sparse=True, **rsc_data)
#     interpolated_east_up = np.empty((2, nrows, ncols))
#     for idx in (0, 1):
#         component = east_up[:, idx]
#         interpolated_east_up[idx] = interpolate.griddata(
#             points=grid_corners, values=component, xi=(xx, yy))
#     interpolated_east_up = interpolated_east_up.reshape((2, nrows * ncols))
