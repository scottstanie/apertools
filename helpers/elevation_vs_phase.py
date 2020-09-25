#!/usr/bin/env python
"""
Makes scatterplots of the phase values vs elevation to see if linear trend exists
"""
import matplotlib.pyplot as plt

# import multiprocessing
# import os
import numpy as np

from apertools import sario


def save_unw_vs_elevation(unw_file_list, every=50):
    points = np.empty((0, 2))
    elpts = sario.load("elevation_looked.dem")
    mask = ~np.isnan(sario.load(unw_file_list[0].replace(".unw", ".unwflat")))
    # elpts = elpts[:, :400]
    elpts = elpts[mask]
    elpts = elpts.reshape((-1,))

    for f in unw_file_list[::every]:
        out = sario.load(f)
        out -= np.mean(out)
        # new_pts = np.vstack((elpts, out[:, :400][mask].reshape((-1, )))).T
        new_pts = np.vstack((elpts, out[mask].reshape((-1,)))).T
        points = np.vstack((points, new_pts))
    np.save("elevation_points.npy", points)


def plot_unw_vs_elevation():
    elpts = np.load("elevation_points.npy")[::1000]
    plt.figure()
    plt.scatter(elpts[:, 0], elpts[:, 1], s=0.1)
    plt.xlabel("elevation")
    plt.ylabel("unwrapped phase (rad)")
    plt.title("sample of all unwrapped igram points vs elevation")
    plt.show(block=True)


if __name__ == "__main__":
    # gdal_translate -outsize 2% 2% S1B_20190325.geo.vrt looked_S1B_20190325.geo.tif
    ifg_date_list = sario.find_igrams(".")
    unw_file_list = [
        f.replace(".int", ".unw") for f in sario.find_igrams(".", parse=False)
    ]
    # unw_file_list = [f.replace(".int", ".unwflat") for f in sario.find_igrams(".", parse=False)]
    save_unw_vs_elevation(unw_file_list)
    plot_unw_vs_elevation()
