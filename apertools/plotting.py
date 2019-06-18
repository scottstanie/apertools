"""plotting.py: functions for visualizing insar products
"""
from __future__ import division, print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from apertools.log import get_log
from apertools import utils, latlon

logger = get_log()


def make_dismph_colormap():
    """Make a custom colormap like the one used in dismph.

    The list was created from dismphN.mat in geodmod which is a 64 segmented colormap
    using the following:
      from scipy.io import loadmat
      cmap = loadmat('dismphN.mat',struct_as_record=True)['dismphN']
      from matplotlib.colors import rgb2hex
      list=[]
      for i in cmap: list.append(rgb2hex(i))

    Source: https://imaging.unavco.org/data/geoslc/geoslc2intf.py
    """
    colors = [
        '#f579cd', '#f67fc6', '#f686bf', '#f68cb9', '#f692b3', '#f698ad', '#f69ea7', '#f6a5a1',
        '#f6ab9a', '#f6b194', '#f6b78e', '#f6bd88', '#f6c482', '#f6ca7b', '#f6d075', '#f6d66f',
        '#f6dc69', '#f6e363', '#efe765', '#e5eb6b', '#dbf071', '#d0f477', '#c8f67d', '#c2f684',
        '#bbf68a', '#b5f690', '#aff696', '#a9f69c', '#a3f6a3', '#9cf6a9', '#96f6af', '#90f6b5',
        '#8af6bb', '#84f6c2', '#7df6c8', '#77f6ce', '#71f6d4', '#6bf6da', '#65f6e0', '#5ef6e7',
        '#58f0ed', '#52e8f3', '#4cdbf9', '#7bccf6', '#82c4f6', '#88bdf6', '#8eb7f6', '#94b1f6',
        '#9aabf6', '#a1a5f6', '#a79ef6', '#ad98f6', '#b392f6', '#b98cf6', '#bf86f6', '#c67ff6',
        '#cc79f6', '#d273f6', '#d86df6', '#de67f6', '#e561f6', '#e967ec', '#ed6de2', '#f173d7'
    ]
    dismphCM = matplotlib.colors.LinearSegmentedColormap.from_list('dismph', colors)
    dismphCM.set_bad('w', 0.0)
    return dismphCM


def discrete_seismic_colors(n=5):
    """From http://colorbrewer2.org/#type=diverging&scheme=RdBu&n=7"""
    if n == 5:
        return list(
            np.array([
                (5, 113, 176),
                (146, 197, 222),
                (247, 247, 247),
                (244, 165, 130),
                (202, 0, 32),
            ]) / 256)
    elif n == 7:
        # Really this is red-yellow-blue
        # http://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=7
        return list(
            np.array([
                (69, 117, 180, 256),
                (145, 191, 219, 256),
                (224, 243, 248, 256),
                (255, 255, 191, 245),  # small alpha drop
                (254, 224, 144, 256),
                (252, 141, 89, 256),
                (215, 48, 39, 256),
            ]) / 256)
        # Using the Red-white-blue
        # return list(
        #     np.array([
        #         (33, 102, 172, 256),
        #         (103, 169, 207, 256),
        #         (209, 229, 240, 256),
        #         (247, 247, 247, 250),
        #         (253, 219, 199, 256),
        #         (239, 138, 98, 256),
        #         (178, 24, 43, 256),
        #     ]) / 256)


DISMPH = make_dismph_colormap()
plt.register_cmap(cmap=DISMPH)
DISCRETE_SEISMIC5 = matplotlib.colors.LinearSegmentedColormap.from_list('discrete_seismic5',
                                                                        discrete_seismic_colors(5),
                                                                        N=5)
plt.register_cmap(cmap=DISCRETE_SEISMIC5)
DISCRETE_SEISMIC7 = matplotlib.colors.LinearSegmentedColormap.from_list('discrete_seismic7',
                                                                        discrete_seismic_colors(7),
                                                                        N=7)
plt.register_cmap(cmap=DISCRETE_SEISMIC7)


def shifted_color_map(cmap, start=0, midpoint=0.5, stop=1.0, num_levels=None):
    """Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Attribution: https://stackoverflow.com/a/20528097, Paul H

    Args:
      cmap (str or matplotlib.cmap): The matplotlib colormap to be altered.
          Can be matplitlib.cm.seismic or 'seismic'
      start (float): Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint (float): The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop (float): Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
      num_levels (int): form fewer discrete levels in the colormap

    Returns:
        matplotlib.cmap
    """
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap, num_levels)

    cdict = {'red': [], 'green': [], 'blue': [], 'alpha': []}

    # regular index to compute the colors
    # N = num_levels
    N = 256
    reg_index = np.linspace(start, stop, N + 1)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, N // 2, endpoint=False),
        np.linspace(midpoint, 1.0, N // 2 + 1, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap('shiftedcmap', cdict, N=num_levels or N)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def make_shifted_cmap(img=None, vmax=None, vmin=None, cmap_name='seismic', num_levels=None):
    """Scales the colorbar so that 0 is always centered (white)"""
    if img is not None:
        if vmin is None:
            vmin = np.nanmin(img)
        if vmax is None:
            vmax = np.nanmax(img)

    if vmax is None or vmin is None:
        raise ValueError("Required args: img, or vmax and vmin")
    midpoint = 1 - vmax / (abs(vmin) + vmax)
    return shifted_color_map(cmap_name, midpoint=midpoint, num_levels=num_levels)


def plot_image_shifted(img,
                       fig=None,
                       ax=None,
                       cmap='seismic',
                       img_data=None,
                       title='',
                       label='',
                       xlabel='',
                       ylabel='',
                       vmin=None,
                       vmax=None,
                       aspect='auto',
                       perform_shift=True,
                       colorbar=True):
    """Plot an image with a zero-shifted colorbar

    Args:
        img (ndarray): 2D numpy array to imshow
        fig (matplotlib.Figure): Figure to plot image onto
        ax (matplotlib.AxesSubplot): Axes to plot image onto
        cmap (str): name of colormap to shift
        img_data (dict): rsc_data from load_dem_rsc containing lat/lon
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
    nrows, ncols = img.shape
    if img_data:
        extent = latlon.grid_extent(**img_data)
    else:
        extent = (0, ncols, nrows, 0)

    if not fig and not ax:
        fig = plt.figure()
        ax = fig.gca()
    elif ax and not fig:
        fig = ax.figure
    elif fig and not ax:
        ax = fig.gca()

    if perform_shift:
        shifted_cmap = make_shifted_cmap(img, cmap_name=cmap, vmin=vmin, vmax=vmax)
        axes_image = ax.imshow(img,
                               cmap=shifted_cmap,
                               extent=extent,
                               vmin=vmin,
                               vmax=vmax,
                               aspect=aspect)
    else:
        vmax = _abs_max(img)
        axes_image = ax.imshow(img, cmap=cmap, extent=extent, vmax=vmax, vmin=-vmax, aspect=aspect)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if colorbar:
        cbar = fig.colorbar(axes_image, ax=ax)
        cbar.set_label(label)
    return fig, axes_image


def _abs_max(img):
    """Find largest absolute value ignoring nans"""
    return np.nanmax(np.abs(img))


def view_stack(
        stack,
        display_img,
        geolist=None,
        label="Centimeters",
        cmap='seismic',
        title='',
        lat_lon=False,
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
        title (str): Optional- Title for plot
        lat_lon (dict): Optional- Uses latitude and longitude in legend
            instead of row/col

    Raises:
        ValueError: if display_img is not an int or the string 'mean'

    """
    # If we don't have dates, use indices as the x-axis
    if geolist is None:
        geolist = np.arange(stack.shape[0])

    imagefig = plt.figure()

    if isinstance(display_img, int):
        img = stack[display_img, :, :]
    elif hasattr(display_img, '__array_finalize__'):
        # TODO: figure out best way to check ndarray or Latlonimage
        img = display_img
    else:
        raise ValueError("display_img must be an int or ndarray-like obj")

    title = title or "Deformation Time Series"  # Default title
    plot_image_shifted(img,
                       fig=imagefig,
                       title=title,
                       cmap='seismic',
                       label=label,
                       perform_shift=False)

    timefig = plt.figure()

    plt.title(title)
    legend_entries = []

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

        if lat_lon:
            lat, lon = img.rowcol_to_latlon(row, col)
            legend_entries.append('Lat {:.3f}, Lon {:.3f}'.format(lat, lon))
        else:
            legend_entries.append('Row %s, Col %s' % (row, col))

        plt.plot(geolist, timeline, marker='o', linestyle='dashed', linewidth=1, markersize=4)
        plt.legend(legend_entries, loc='upper left')
        x_axis_str = "SAR image date" if geolist is not None else "Image number"
        plt.xlabel(x_axis_str)
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
