"""plotting.py: functions for visualizing insar products
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from apertools.log import get_log
from apertools import utils, latlon
from .colors import make_shifted_cmap

try:
    basestring = str
except NameError:
    pass

logger = get_log()


def get_fig_ax(fig, ax):
    """Handle passing None to either fig or ax by creating new, returns both"""
    if not fig and not ax:
        fig = plt.figure()
        ax = fig.gca()
    elif ax and not fig:
        fig = ax.figure
    elif fig and not ax:
        ax = fig.gca()
    return fig, ax


def plot_image(img=None,
               filename=None,
               dset=None,
               fig=None,
               ax=None,
               cmap='seismic_wide_y',
               rsc_data=None,
               title='',
               label='',
               xlabel='',
               ylabel='',
               vm=None,
               twoway=True,
               vmin=None,
               vmax=None,
               aspect='auto',
               perform_shift=False,
               colorbar=True):
    """Plot an image with a zero-shifted colorbar

    Args:
        img (ndarray): 2D numpy array to imshow
        filename (str): name of file to load image
        dset (str): name HDF5 dataset for `filename` loading
        fig (matplotlib.Figure): Figure to plot image onto
        ax (matplotlib.AxesSubplot): Axes to plot image onto
        cmap (str): name of colormap to shift
        rsc_data (dict): rsc_data from load_dem_rsc containing lat/lon
            data about image, used to make axes into lat/lon instead of row/col
        title (str): Title for image
        label (str): label for colorbar
        aspect (str): passed to imshow aspect
            see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html
        perform_shift (bool): default True. If false, skip cmap shifting step
        colorbar (bool): display colorbar in figure
    Returns:
        fig, axes_image
    """
    if filename and dset:
        from apertools import sario
        import h5py
        with h5py.File(filename, "r") as f:
            img = f[dset][:]
        rsc_data = sario.load_dem_from_h5(filename)

    nrows, ncols = img.shape
    if rsc_data:
        extent = latlon.grid_extent(**rsc_data)
    else:
        extent = (0, ncols, nrows, 0)

    fig, ax = get_fig_ax(fig, ax)
    if not fig and not ax:
        fig = plt.figure()
        ax = fig.gca()
    elif ax and not fig:
        fig = ax.figure
    elif fig and not ax:
        ax = fig.gca()

    if perform_shift:
        shifted_cmap = make_shifted_cmap(img, cmap_name=cmap, vmin=vmin, vmax=vmax)
        axes_image = ax.imshow(
            img,
            cmap=shifted_cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
        )
    else:
        vmin, vmax = _get_vminmax(img, vm=vm, vmin=vmin, vmax=vmax, twoway=twoway)
        axes_image = ax.imshow(
            img,
            cmap=cmap,
            extent=extent,
            vmax=vmax,
            vmin=vmin,
            aspect=aspect,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if colorbar:
        cbar = fig.colorbar(axes_image, ax=ax)
        cbar.set_label(label)
    plt.show(block=False)
    return fig, axes_image


def _abs_max(img):
    """Find largest absolute value ignoring nans"""
    return np.nanmax(np.abs(img))


def _get_vminmax(img, vm=None, vmin=None, vmax=None, twoway=True):
    img_nonan = img[~np.isnan(img)]
    vm = vm or np.max(np.abs(img_nonan))
    if twoway:
        vmax = vm if vmax is None else vmax
        vmin = -vm if vmin is None else vmin
    else:
        vmax = vm if vmax is None else vmax
        vmin = 0 if vmin is None else vmin
    return vmin, vmax


def view_stack(
    stack,
    display_img,
    geolist=None,
    label="Centimeters",
    cmap='seismic_wide',
    perform_shift=False,
    title='',
    vmin=None,
    vmax=None,
    legend_loc="upper left",
    lat_lon=False,
    line_plot_kwargs=None,
    timeline_callback=None,
):
    """Displays an image from a stack, allows you to click for timeseries

    Args:
        stack (ndarray): 3D np.ndarray, 1st index is image number
            i.e. the idx image is stack[idx, :, :]
        geolist (list[datetime]): Optional: times of acquisition for
            each stack layer. Used as xaxis if provided
        display_img (LatlonImage): Give which image in the stack you want as the display
        label (str): Optional- Label on colorbar/yaxis for plot
            Default = Centimeters
        cmap (str): Optional- colormap to display stack image (default='seismic')
        perform_shift (bool): default False, make the colormap shift to center at 0
            e.g. When False, will make  colorbar -10 to 10 if data is within [-1,10]
            but when True, makes it tight to [-1, 10]
        title (str): Optional- Title for plot
        legend_loc (str): Default 'upper left', for the line plot
        lat_lon (dict): Optional- Uses latitude and longitude in legend
            instead of row/col
        line_plot_kwargs (dict): matplotlib options for the line plot
        timeline_callback (func): callback to run on a pixels timeseries
            Must have signature: func(timeseries, row, col)
            e.g.: timeline_callback=lambda t: print(t, row, col)

    Raises:
        ValueError: if display_img is not an int or the string 'mean'

    """
    # If we don't have dates, use indices as the x-axis
    if geolist is None:
        geolist = np.arange(stack.shape[0])

    imagefig = plt.figure()

    if isinstance(display_img, int):
        img = stack[display_img, :, :]
    # TODO: is this the best way to check if it's ndarray or Latlonimage?
    elif hasattr(display_img, '__array_finalize__'):
        img = display_img
    else:
        raise ValueError("display_img must be an int or ndarray-like obj")

    title = title or "Deformation Time Series"  # Default title
    plot_image(img,
                       fig=imagefig,
                       title=title,
                       cmap=cmap,
                       label=label,
                       vmin=vmin,
                       vmax=vmax,
                       perform_shift=perform_shift)

    timefig = plt.figure()

    plt.title(title)
    legend_entries = []
    if not line_plot_kwargs:
        line_plot_kwargs = dict(marker='o', linestyle='dashed', linewidth=1, markersize=4)

    def onclick(event):
        # Ignore right/middle click, clicks off image
        if event.button != 1 or not event.inaxes:
            return
        # Check if the toolbar has zoom or pan active
        # https://stackoverflow.com/a/20712813
        if imagefig.canvas.manager.toolbar._active is not None:
            return
        plt.figure(timefig.number)
        row, col = int(event.ydata), int(event.xdata)
        # Somehow clicked outside image, but in axis
        if row >= img.shape[0] or col >= img.shape[1]:
            return
        timeline = stack[:, row, col]

        if timeline_callback is not None:
            timeline_callback(timeline, row, col)

        if lat_lon:
            lat, lon = img.rowcol_to_latlon(row, col)
            legend_entries.append('Lat {:.3f}, Lon {:.3f}'.format(lat, lon))
        else:
            legend_entries.append('Row %s, Col %s' % (row, col))

        plt.figure(2)
        plt.plot(geolist, timeline, **line_plot_kwargs)
        plt.legend(legend_entries, loc=legend_loc)
        x_axis_str = "SAR image date" if geolist is not None else "Image number"
        plt.xlabel(x_axis_str)
        timefig.autofmt_xdate()
        plt.ylabel(label)
        plt.show()

    imagefig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)


def equalize_and_mask(image, low=1e-6, high=2, fill_value=np.inf, db=True):
    """Clips an image to increase contrast"""
    # Mask the invalids, then mask zeros, then clip rest
    im = np.clip(utils.mask_zeros(np.ma.masked_invalid(image)), low, high)
    if fill_value:
        im.set_fill_value(fill_value)
    return utils.db(im) if db else im


def animate_stack(stack,
                  pause_time=200,
                  display=True,
                  titles=None,
                  label=None,
                  save_title=None,
                  cmap_name='seismic',
                  shifted=True,
                  vmin=None,
                  vmax=None,
                  **savekwargs):
    """Runs a matplotlib loop to show each image in a 3D stack

    Args:
        stack (ndarray): 3D np.ndarray, 1st index is image number
            i.e. the idx image is stack[idx, :, :]
        pause_time (float): Optional- time between images in milliseconds (default=200)
        display (bool): True if you want the plot GUI to pop up and run
            False would be if you jsut want to save the movie with as save_title
        titles (list[str]): Optional- Names of images corresponding to stack.
            Length must match stack's 1st dimension length
        label (str): Optional- Label for the colorbar
        save_title (str): Optional- if provided, will save the animation to a file
            extension must be a valid extension for a animation writer:
        cmap_name (str): Name of matplotlib colormap
        shifted (bool): default true: shift the colormap to be 0 centered
        vmin (float): min value passed to imshow
        vmax (float): max value passed to imshow
        savekwargs: extra keyword args passed to animation.save
            See https://matplotlib.org/api/_as_gen/matplotlib.animation.Animation.html
            and https://matplotlib.org/api/animation_api.html#writer-classes

    Returns:
        stack_ani: the animation object
    """
    num_images = stack.shape[0]
    if titles:
        assert len(titles) == num_images, "len(titles) must equal stack.shape[0]"
    else:
        titles = ['' for _ in range(num_images)]  # blank titles, same length
    if np.iscomplexobj(stack):
        stack = np.abs(stack)

    # Use the same stack min and stack max (or vmin/vmax) for all colorbars/ color ranges
    vmin = np.min(stack) if vmin is None else vmin
    vmax = np.max(stack) if vmax is None else vmax
    cmap = cmap_name if not shifted else make_shifted_cmap(
        vmin=vmin, vmax=vmax, cmap_name=cmap_name)

    fig, ax = plt.subplots()
    axes_image = plt.imshow(stack[0, :, :], vmin=vmin, vmax=vmax, cmap=cmap)

    cbar = fig.colorbar(axes_image)
    cbar_ticks = np.linspace(vmin, vmax, num=6, endpoint=True)
    cbar.set_ticks(cbar_ticks)
    if label:
        cbar.set_label(label)

    def update_im(idx):
        axes_image.set_data(stack[idx, :, :])
        fig.suptitle(titles[idx])
        return axes_image,

    stack_ani = animation.FuncAnimation(fig,
                                        update_im,
                                        frames=range(num_images),
                                        interval=pause_time,
                                        blit=False,
                                        repeat=True)

    if save_title:
        logger.info("Saving to %s", save_title)
        stack_ani.save(save_title, writer='imagemagick', **savekwargs)

    if display:
        plt.show()
    return stack_ani


def make_figure_noborder():
    fig = plt.figure(frameon=False)

    # To make the content fill the whole figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax


def set_aspect_image(fig, img, height=4):
    """Adjusts sizes to match image data ratio

    Can pick a height (default 4), and keeps data rows/col ratio
    """
    nrows, ncols = img.shape
    width = ncols / nrows * height
    print('width', width, 'height', height)
    fig.set_size_inches(width, height)


def save_paper_figure(fig, fname, axis_off=True):
    fig.tight_layout()
    if axis_off:
        plt.axis('off')
    print('Saving %s' % fname)
    fig.savefig(fname, bbox_inches='tight', transparent=True, dpi=300)


def plot_shapefile(filename, fig=None, ax=None, z=None):
    # Credit: https://gis.stackexchange.com/a/152331
    import shapefile
    fig, ax = get_fig_ax(fig, ax)

    with shapefile.Reader(filename) as sf:
        for shape in sf.shapeRecords():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            if z is not None:
                ax.plot3D(x, y, z, 'b')
            else:
                ax.plot(x, y, 'b')

        plt.show()


# def plotcompare(fnames, dset="velos", vmax=25, vmin=-25, cmap="seismic_wide", **kwargs):
def plot_img_diff(arrays=None,
                  dset="velos/1",
                  fnames=[],
                  vm=20,
                  vdiff=4,
                  vmax=None,
                  vmin=None,
                  twoway=True,
                  cmap="seismic_wide_y",
                  show=True,
                  **kwargs):
    """Compare two images and their difference"""
    if arrays is None:
        from apertools import sario
        arrays = [sario.load(f, dset=dset, **kwargs) for f in fnames]

    n = len(arrays)
    vmin, vmax = _get_vminmax(arrays[0], vm=vm, vmin=vmin, vmax=vmax, twoway=twoway)
    print(f"{vmin} {vmax}")
    fig, axes = plt.subplots(1, n+1, sharex=True, sharey=True)
    for ii in range(n):
        axim = axes[ii].imshow(arrays[ii], cmap=cmap, vmax=vmax, vmin=vmin)
    fig.colorbar(axim, ax=axes[-2])
    # Now different image at end
    axim = axes[-1].imshow(arrays[0] - arrays[1], cmap=cmap, vmax=vdiff, vmin=-vdiff)
    axes[-1].set_title("left - middle")
    fig.colorbar(axim, ax=axes[-1])
    # [f.close() for f in files]
    if show:
        plt.show(block=False)
    return fig, axes
