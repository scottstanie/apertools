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
from apertools.utils import filter_min_max_date


def stitch_same_dates(
    geo_path=".", output_path=".", reverse=True, overwrite=False, verbose=True
):
    """Combines .geo files of the same date in one directory

    The reverse argument is to specify which order the geos get sorted.
    If reverse=True, then the later geo is used in the overlapping strip.
    This seems to work better for some descending path examples.
    """
    grouped_geos = group_geos_by_date(geo_path, reverse=reverse)

    for _, slclist in grouped_geos:
        stitch_geos(
            slclist,
            reverse,
            output_path,
            overwrite=overwrite,
            verbose=verbose,
        )

    return grouped_geos


def group_geos_by_date(geo_path, reverse=True, ext=".geo"):
    def _make_groupby(slclist):
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
        return [
            (date, list(g))
            for date, g in itertools.groupby(slclist, key=lambda x: x.date)
        ]

    # Assuming only IW products are used (included IW to differentiate from my date-only naming)
    geos = [
        parsers.Sentinel(g) for g in glob.glob(os.path.join(geo_path, f"S1*IW*{ext}"))
    ]
    # Find the dates that have multiple frames/.geos
    date_counts = collections.Counter([g.date for g in geos])
    dates_duped = set([date for date, count in date_counts.items() if count > 1])

    double_geo_files = sorted(
        (g for g in geos if g.date in dates_duped),
        key=lambda g: g.start_time,
        reverse=reverse,
    )

    # Now collapse into groups, sorted by the date
    grouped_geos = _make_groupby(double_geo_files)
    return grouped_geos


def stitch_geos(slclist, reverse, output_path, overwrite=False, verbose=True):
    """Combines multiple .geo files of the same date into one image"""
    if verbose:
        print("Stitching geos for %s" % slclist[0].date)
        print("reverse=", reverse)
        for g in slclist:
            print("image:", g.filename, g.start_time)

    g = slclist[0]
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
    # stitched_img = combine_complex([sario.load(g.filename) for g in slclist])
    stitched_img = combine_complex([g.filename for g in slclist])

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
    img1 = (
        img_list[0] if isinstance(img_list[0], np.ndarray) else sario.load(img_list[0])
    )
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
            raise ValueError(
                "All images must have same size. Sizes: %s, %s"
                % (img_shape, next_img.shape)
            )
        nonzero_mask = next_img != 0
        img_out[nonzero_mask] = next_img[nonzero_mask]

        # OLD WAY:
        # img_out += next_img
        # Now only on overlap, take the previous's pixels
        # overlap_idxs = (img_out != 0) & (next_img != 0)
        # img_out[overlap_idxs] = next_img[overlap_idxs]

    return img_out


def find_row_overlaps(ftop, fbot):
    import rasterio as rio

    with rio.open(ftop) as srctop, rio.open(fbot) as srcbot:
        xstep, _, xfirst, _, ystep, yfirst = srctop.transform[:6]
        xstep0, _, xfirst0, _, ystep0, yfirst0 = srcbot.transform[:6]

        topfirst_mid = yfirst + ystep / 2
        toplast_mid = topfirst_mid + (srctop.shape[0] - 1) * ystep

        botfirst_mid = yfirst0 + ystep0 / 2

        row_of_toplast_in_bottom = (toplast_mid - botfirst_mid) / ystep0
        row_of_botfirst_in_top = (botfirst_mid - topfirst_mid) / ystep0
        return row_of_toplast_in_bottom, row_of_botfirst_in_top


def find_rows(ftop, fbot):
    import rasterio as rio

    with rio.open(ftop) as srctop, rio.open(fbot) as srcbot:
        top_lastrow_xy = srctop.xy(*(np.array(srctop.shape) - 1))
        rowcol_of_toplast_in_bottom = srcbot.index(*top_lastrow_xy)

        rowcol_of_botfirst_in_top = srctop.index(*srcbot.xy(0, 0))
        return rowcol_of_toplast_in_bottom, rowcol_of_botfirst_in_top


def rewrap_to_2pi(phase):
    return np.mod(phase + np.pi, 2 * np.pi) - np.pi


def stitch_topstrip(ftop, fbot, colshift=1):
    # This worked:
    # Out[14]: ((540, 32400), (2160, 0))
    # In [29]: out = np.zeros((len(gbot) + len(gtop) - 540, gtop.shape[1]), dtype=gtop.dtype)
    # In [30]: out[:len(gtop), 1:] = gtop[:, 1:]
    # In [31]: out[-len(gbot):] = gbot

    print(f"Stitching {ftop} and {fbot}")
    (toplast_in_bot, _), (botfirst_in_top, _) = find_rows(ftop, fbot)
    print(f"{toplast_in_bot = }")
    # (botfirst_in_top, _), (toplast_in_bot, _) = find_rows(ftop, fbot)
    # breakpoint()
    imgtop = sario.load(ftop, band=1)
    # print("redo shift", imgtop[:5, -5:])
    imgbot = sario.load(fbot, band=1)

    if np.iscomplexobj(imgtop):
        # Find any constant phase offset in overlap
        # np.abs(gbot[:361, -499:]), np.abs(gtop[1440:, -500:-1])
        bot_slice = imgbot[: toplast_in_bot + 1, colshift:]
        top_slice = imgtop[botfirst_in_top:, :-colshift]

        phase_diff = rewrap_to_2pi(np.angle(top_slice) - np.angle(bot_slice))
        # phase_shift = phase_diff.mean(axis=0)
        phase_shift = phase_diff.mean()
        imgtop[:, :-colshift] = imgtop[:, :-colshift] * np.exp(1j * phase_shift)

    total_rows = imgbot.shape[0] + imgtop.shape[0] - toplast_in_bot

    out = np.zeros((total_rows, imgbot.shape[1]), dtype=imgbot.dtype)
    out[-len(imgbot) :] = imgbot
    # out[:len(imgtop), colshift:] = imgtop[:, colshift:]
    # out[:len(imgtop), colshift:] = imgtop[:, :-colshift]
    out[: len(imgtop), :-colshift] = imgtop[:, colshift:]
    return out

    # botrows = imgbot.shape[0]
    # toprows = imgtop.shape[0]
    # # out[:botfirst_in_top + 1] = imgtop[:botfirst_in_top + 1]
    # # out[:toplast_in_bot] = imgtop[:toplast_in_bot]
    # out[:toprows] = imgtop[:toprows]
    # out[-botrows:] = imgbot
    # return out


def test_ifg(d1, d2, looks=30):
    import apertools.utils
    g1 = stitch_topstrip(f'../top_strip/S1A_{d1}.geo.vrt', f'../S1A_{d1}.geo.vrt')
    g2 = stitch_topstrip(f'../top_strip/S1A_{d2}.geo.vrt', f'../S1A_{d2}.geo.vrt')
    nr = 3000
    return apertools.utils.take_looks(g1[:nr] * g2[:nr].conj(), looks, looks)