#!/usr/bin/env python
import argparse
# import os
# import glob

import matplotlib.pyplot as plt
import numpy as np
import apertools.sario as sario
import apertools.utils as utils


def plot_image(img, title=None, colorbar=True, alpha=0.6):
    if np.iscomplexobj(img):
        img_abs = np.abs(img)
        img_phase = np.angle(img)
    else:
        img_abs = img
        img_phase = None

    high = np.percentile(img_abs.reshape(-1), 99)

    # low = np.percentile(np.abs(img.reshape(-1)), 5)
    low = 5  # Found to look good to clip radar magnitudes to 5 on low end

    img_abs = np.clip(img_abs, low, high)
    img_db = utils.db(img_abs)

    fig, axes = plt.subplots()
    axim = axes.imshow(img_db, cmap='gray')
    if img_phase is not None:
        axes.imshow(img_phase, cmap='dismph', alpha=alpha)

    if colorbar:
        fig.colorbar(axim)

    if title:
        axes.set_title(title)

    print('h')
    plt.show(block=True)
    print('i')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Name of file to open")
    parser.add_argument("-d",
                        "--downsample",
                        type=int,
                        default=1,
                        help="Factor to downsample file to display (default=1)")
    parser.add_argument("--dem-rsc", help="Name of dem.rsc file to use for opening image")
    parser.add_argument("--colorbar", action="store_true", default=True, help="Show colorbar")
    parser.add_argument("--title", help="Title for figure")
    args = parser.parse_args()

    img = sario.load(args.filename, downsample=args.downsample)
    plot_image(img, title=args.title, colorbar=args.colorbar)
