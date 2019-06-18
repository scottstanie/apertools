#!/usr/bin/env python
"""stitching.py: utilities for slicing and combining images
"""
from __future__ import division, print_function
import collections
import itertools
import glob
import os

from apertools import parsers, sario, utils


def stitch_same_dates(geo_path=".", output_path=".", reverse=True):
    """Combines .geo files of the same date in one directory

    The reverse argument is to specify which order the geos get sorted.
    If reverse=True, then the later geo is used in the overlapping strip.
    This seems to work better for some descending path examples.
    """

    def _group_geos_by_date(geolist):
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
    grouped_geos = _group_geos_by_date(double_geo_files)
    for date, geolist in grouped_geos:
        print("Stitching geos for %s" % date)
        # TODO: Make combine handle more than 2!
        g1, g2 = geolist[:2]

        print('reverse=', reverse)
        print('image 1:', g1.filename, g1.start_time)
        print('image 2:', g2.filename, g2.start_time)
        stitched_img = utils.combine_complex(
            sario.load(g1.filename),
            sario.load(g2.filename),
        )
        new_name = "{}_{}.geo".format(g1.mission, g1.date.strftime("%Y%m%d"))
        new_name = os.path.join(output_path, new_name)
        print("Saving stitched to %s" % new_name)
        # Remove any file with same name before saving
        # This prevents symlink overwriting old files
        utils.rm_if_exists(new_name)
        sario.save(new_name, stitched_img)

    return grouped_geos
