#!/usr/bin/env python
"""stitching.py: utilities for slicing and combining images
"""
from __future__ import division, print_function
import collections
import itertools
import glob
import os
import numpy as np

from apertools import parsers, sario, utils


def stitch_same_dates(geo_path=".", output_path=".", reverse=True, verbose=True):
    """Combines .geo files of the same date in one directory

    The reverse argument is to specify which order the geos get sorted.
    If reverse=True, then the later geo is used in the overlapping strip.
    This seems to work better for some descending path examples.
    """
    grouped_geos = group_geos_by_date(geo_path, reverse=reverse)

    for date, geolist in grouped_geos:
        stitch_geos(date, geolist, reverse, output_path, verbose)

    return grouped_geos


def group_geos_by_date(geo_path, reverse=True):
    def _make_groupby(geolist):
        """Groups into sub-lists sharing dates
        example input:
        [Sentinel S1B, path 78 from 2017-10-13,
         Sentinel S1B, path 78 from 2017-10-13,
         Sentinel S1B, path 78 from 2017-10-25,
         Sentinel S1B, path 78 from 2017-10-25]

        Output:
        [(datetime.date(2017, 10, 13),
          [Sentinel S1B, path 78 from 2017-10-13,
           Sentinel S1B, path 78 from 2017-10-13]),
         (datetime.date(2017, 10, 25),
          [Sentinel S1B, path 78 from 2017-10-25,
           Sentinel S1B, path 78 from 2017-10-25])]

        """
        return [(date, list(g)) for date, g in itertools.groupby(geolist, key=lambda x: x.date)]

    geos = [parsers.Sentinel(g) for g in glob.glob(os.path.join(geo_path, "S1*SLC*.geo"))]
    # Find the dates that have multiple frames/.geos
    date_counts = collections.Counter([g.date for g in geos])
    dates_duped = set([date for date, count in date_counts.items() if count > 1])

    double_geo_files = sorted((g for g in geos if g.date in dates_duped),
                              key=lambda g: g.start_time,
                              reverse=reverse)

    grouped_geos = _make_groupby(double_geo_files)
    return grouped_geos


def stitch_geos(date, geolist, reverse, output_path, verbose=True):
    """Combines multiple .geo files of the same date into one image"""
    if verbose:
        print("Stitching geos for %s" % date)
        print('reverse=', reverse)
        for g in geolist:
            print('image:', g.filename, g.start_time)

    # TODO: load in parallel
    # TODO: stitch in paralle;
    stitched_img = combine_complex([sario.load(g.filename) for g in geolist])

    g = geolist[0]
    new_name = "{}_{}.geo".format(g.mission, g.date.strftime("%Y%m%d"))
    new_name = os.path.join(output_path, new_name)
    print("Saving stitched to %s" % new_name)
    # Remove any file with same name before saving
    # This prevents symlink overwriting old files
    utils.rm_if_exists(new_name)
    sario.save(new_name, stitched_img)


def combine_complex(img_list):
    """Combine multiple complex images which partially overlap

    Used for SLCs/.geos of adjacent Sentinel frames

    Args:
        img_list (ndarray): list of complex images (.geo files)
    Returns:
        ndarray: Same size as each, with pixels combined
    """
    if len(img_list) < 2:
        raise ValueError("Must pass more than 1 image to combine")
    img_shapes = set([img.shape for img in img_list])
    if len(img_shapes) > 1:
        raise ValueError("All images must have same size. Sizes: %s" % ','.join(img_shapes))
    # Start with each one where the other is nonzero
    img1 = img_list[0]

    img_out = np.copy(img1)
    for next_img in img_list[1:]:
        img_out += next_img
        # Now only on overlap, take the previous's pixels
        overlap_idxs = (img_out != 0) & (next_img != 0)
        img_out[overlap_idxs] = img_out[overlap_idxs]

    return img_out
