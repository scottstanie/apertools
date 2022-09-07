from os import fspath

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from osgeo import gdal


def plot(
    nmap_filename,
    slc_stack_filename=None,
    slc_bands=[1, 2, 3],
    slc_filename=None,
    alpha_nmap=0.8,
    cmap_nmap="Reds_r",
    block=False,
):
    """Interactively plot the neighborhood map over an SLC image.

    Click on a pixel to see the neighborhood map.
    """
    slc = _load_slc(slc_stack_filename, slc_bands, slc_filename)

    fig, ax = plt.subplots()
    axim_slc = ax.imshow(_scale_mag(slc), cmap="gray", vmax=np.percentile(slc, 99))

    ny, nx = _get_windows(nmap_filename)

    def onclick(event):
        # Ignore right/middle click, clicks off image
        if event.button != 1 or not event.inaxes:
            return
        # Check if the toolbar has zoom or pan active
        # https://stackoverflow.com/a/20712813
        # MPL version 3.3: https://stackoverflow.com/a/63447351
        # if mpl.__version__ >= "3.3":
        state = fig.canvas.manager.toolbar.mode
        if state != "":  # Zoom/other tool is active
            return

        # Save limits to restore after adding neighborhoods
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        row, col = int(event.ydata), int(event.xdata)
        # Somehow clicked outside image, but in axis
        if row >= slc.shape[0] or col >= slc.shape[1]:
            return

        n = load_neighborhood(nmap_filename, row, col)
        n_img = np.ma.masked_where(n == 0, n)
        extent = _get_extent(row, col, ny, nx)

        # Remove old neighborhood images
        [img.remove() for img in event.inaxes.get_images() if img != axim_slc]

        axim_nmap = ax.imshow(
            n_img,
            cmap=cmap_nmap,
            alpha=alpha_nmap,
            extent=extent,
            origin="lower",
        )
        # Remove old neighborhood patches
        [p.remove() for p in ax.patches]
        # add a rectangle around the neighborhood
        rect = patches.Rectangle(
            (extent[0], extent[2]),
            1 + 2 * nx,
            1 + 2 * ny,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Restore original viewing bounds
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show(block=block)


def load_neighborhood(filename, row, col):
    """Get the neighborhood of a pixel as a numpy array

    Parameters
    ----------
    row : int
        Row of the pixel
    col: int
        column of pixel

    Returns
    -------
    neighborhood : numpy array (dtype = np.bool)
    """

    ny, nx = _get_windows(filename)
    # nbands = ds.RasterCount
    # Shape: (nbands, 1, 1)
    ds = gdal.Open(fspath(filename), gdal.GA_ReadOnly)
    pixel = ds.ReadAsArray(xoff=col, yoff=row, xsize=1, ysize=1)
    ds = None

    pixel = pixel.ravel()
    assert pixel.view("uint8").shape[0] == 4 * len(pixel)
    # 1D version of neighborhood, padded with zeros
    neighborhood = np.unpackbits(pixel.view("uint8"), bitorder="little")
    wx = 2 * nx + 1
    wy = 2 * ny + 1
    ntotal = wx * wy
    assert np.all(neighborhood[ntotal:] == 0)
    return neighborhood[:ntotal].reshape((wy, wx))


def _get_windows(filename):
    ds = gdal.Open(fspath(filename), gdal.GA_ReadOnly)
    if "ENVI" in ds.GetMetadataDomainList():
        meta = ds.GetMetadata("ENVI")
    else:
        meta = ds.GetMetadata()
    nx = int(meta["HALFWINDOWX"])
    ny = int(meta["HALFWINDOWY"])
    ds = None
    return ny, nx


def _get_extent(row, col, ny, nx):
    """Get the row/col extent of the window surrounding a pixel."""
    # Matplotlib extent is (left, right, bottom, top)
    # Also the extent for normal `imshow` is shifted by -0.5
    return col - nx - 0.5, col + nx + 1 - 0.5, row - ny - 0.5, row + ny + 1 - 0.5


def _load_slc(slc_stack_filename, stack_bands, slc_filename):
    if slc_stack_filename is not None:
        ds = gdal.Open(fspath(slc_stack_filename), gdal.GA_ReadOnly)
        # Average the power of the complex bands
        slc = np.zeros((ds.RasterYSize, ds.RasterXSize), dtype=np.float32)
        for b in stack_bands:
            slc += np.abs(ds.GetRasterBand(b).ReadAsArray()) ** 2
        slc = np.sqrt(slc / len(stack_bands))
        ds = None
    else:
        # If one SLC is provided, use that
        ds = gdal.Open(fspath(slc_filename), gdal.GA_ReadOnly)
        slc = np.abs(ds.ReadAsArray())
        ds = None
    return slc


def _scale_mag(img, exponent=0.3, max_pct=99.95):
    """Scale the magnitude of complex radar image for better display"""
    out = np.abs(img) ** exponent
    max_val = np.nanpercentile(out, max_pct)
    return np.clip(out, None, max_val)
