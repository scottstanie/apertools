#!/usr/bin/env python
"""stitching.py: utilities for slicing and combining images
"""
from __future__ import division, print_function
import collections
import itertools
import glob
import os
import numpy as np

from apertools import parsers, sario


def stitch_same_dates(geo_path=".", output_path=".", reverse=True, overwrite=False, verbose=True):
    """Combines .geo files of the same date in one directory

    The reverse argument is to specify which order the geos get sorted.
    If reverse=True, then the later geo is used in the overlapping strip.
    This seems to work better for some descending path examples.
    """
    grouped_geos = group_geos_by_date(geo_path, reverse=reverse)

    for _, geolist in grouped_geos:
        stitch_geos(
            geolist,
            reverse,
            output_path,
            overwrite=overwrite,
            verbose=verbose,
        )

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

    # Assuming only IW products are used (included IW to differentiate from my date-only naming)
    geos = [parsers.Sentinel(g) for g in glob.glob(os.path.join(geo_path, "S1*IW*.geo"))]
    # Find the dates that have multiple frames/.geos
    date_counts = collections.Counter([g.date for g in geos])
    dates_duped = set([date for date, count in date_counts.items() if count > 1])

    double_geo_files = sorted((g for g in geos if g.date in dates_duped),
                              key=lambda g: g.start_time,
                              reverse=reverse)

    # Now collapse into groups, sorted by the date
    grouped_geos = _make_groupby(double_geo_files)
    return grouped_geos


def stitch_geos(geolist, reverse, output_path, overwrite=False, verbose=True):
    """Combines multiple .geo files of the same date into one image"""
    if verbose:
        print("Stitching geos for %s" % geolist[0].date)
        print('reverse=', reverse)
        for g in geolist:
            print('image:', g.filename, g.start_time)

    g = geolist[0]
    new_name = "{}_{}.geo".format(g.mission, g.date.strftime("%Y%m%d"))
    new_name = os.path.join(output_path, new_name)
    if os.path.exists(new_name):
        if os.path.islink(new_name):
            print("Removing symlink %s" % new_name)
            os.remove(new_name)
        elif overwrite:  # real file
            print("Overwrite=True: Removing %s" % new_name)
            os.remove(new_name)
        else:
            print(" %s exists, not overwriting. skipping" % new_name)
            return

    # TODO: load as blocks, not all at once
    # stitched_img = combine_complex([sario.load(g.filename) for g in geolist])
    stitched_img = combine_complex([g.filename for g in geolist])

    print("Saving stitched to %s" % new_name)
    # Remove any file with same name before saving
    # This prevents symlink overwriting old files
    sario.save(new_name, stitched_img)


def combine_complex(img_list, verbose=True):
    """Combine multiple complex images which partially overlap

    Used for SLCs/.geos of adjacent Sentinel frames

    Args:
        img_list: list of complex images (.geo files)
            can be filenames or preloaded arrays
    Returns:
        ndarray: Same size as each, with pixels combined
    """
    if len(img_list) < 2:
        raise ValueError("Must pass more than 1 image to combine")
    # Start with each one where the other is nonzero
    img1 = img_list[0] if isinstance(img_list[0], np.ndarray) else sario.load(img_list[0])
    img_shape = img1.shape

    total = len(img_list)
    print("processing image 1 of %s" % (total))

    img_out = np.copy(img1)
    for (idx, next_img) in enumerate(img_list[1:]):
        if verbose:
            print("processing image %s of %s" % (idx + 2, total))
        if not isinstance(next_img, np.ndarray):
            next_img = sario.load(next_img)

        if next_img.shape != img_shape:
            raise ValueError("All images must have same size. Sizes: %s, %s" %
                             (img_shape, next_img.shape))
        nonzero_mask = next_img != 0
        img_out[nonzero_mask] = next_img[nonzero_mask]

        # OLD WAY:
        # img_out += next_img
        # Now only on overlap, take the previous's pixels
        # overlap_idxs = (img_out != 0) & (next_img != 0)
        # img_out[overlap_idxs] = next_img[overlap_idxs]

    return img_out
