"""plotting.py: functions for visualizing insar products"""

from math import ceil, sqrt

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import numpy as np


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import matplotlib.dates as mdates
from apertools.log import get_log
from apertools import utils, latlon
from apertools.sario import slclist_to_str
from .colors import make_shifted_cmap, make_dismph_colors

try:
    basestring = str
except NameError:
    pass

logger = get_log()

DEFAULT_CMAP = "seismic_wide_y_r"


class DateSlider(Slider):
    # https://matplotlib.org/stable/_modules/matplotlib/widgets.html#Slider
    def _format(self, val):
        """Pretty-print *val*."""
        if self.valfmt is not None:
            return mdates.num2date(val).strftime(self.valfmt)
        else:
            return mdates.num2date(val).strftime("%Y-%m-%d")


def get_style(size=15, grid_on=False, cmap="viridis", weight="bold", minor_ticks=False):
    style_dict = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # "font.family": "Helvetica",
        "font.family": "sans-serif",
        "font.size": size,
        "font.weight": weight,
        "legend.fontsize": "large",
        "axes.labelsize": size,
        "axes.titlesize": size,
        "xtick.labelsize": size * 0.75,
        "ytick.labelsize": size * 0.75,
        "axes.grid": grid_on,
        "image.cmap": cmap,
        "xtick.minor.visible": minor_ticks,
        "ytick.minor.visible": minor_ticks,
    }
    return style_dict


def set_style(
    size=15,
    nolatex=True,
    grid_on=False,
    cmap="viridis",
    weight="bold",
    minor_ticks=False,
):
    import scienceplots  # noqa: F401

    # As of version 2.0.0, you need to add import scienceplots before setting the style
    style = ["science", "no-latex"] if nolatex else "science"
    plt.style.use(style)
    style_dict = get_style(
        size, grid_on=grid_on, cmap=cmap, weight=weight, minor_ticks=minor_ticks
    )
    plt.rcParams.update(style_dict)
    try:
        import xarray as xr

        xr.set_options(cmap_divergent="RdBu")
    except:
        pass


def scale_mag(img, expval=0.3, max_pct=99.95):
    """Scale the magnitude of complex radar image for better display"""
    out = np.abs(img) ** expval
    max_val = np.nanpercentile(out, max_pct)
    return np.clip(out, None, max_val)


def phase_to_2pi(img):
    """Convert [-pi, pi] phase to [0, 2pi] by adding 2pi to negative vals"""
    phase = np.angle(img) if np.iscomplexobj(img) else img
    if np.nanmin(phase) >= 0 and np.nanmax(phase) <= 2 * np.pi:
        return phase
    assert np.nanmin(phase) >= -3.142 and np.nanmax(phase) <= 3.142
    return np.where(phase > 0, phase, phase + 2 * np.pi)


def make_mph_image(ifg, expval=0.4, max_pct=99, scalemag=True):
    """Convert an ifg array into a PIL.Image with same colors as `dismph`"""
    from PIL import Image

    phase = phase_to_2pi(ifg)
    # Convert the phase values from [0, 2pi] to [0, 359] lookup values
    phase_idxs = (phase / 2 / np.pi * 360 - 0.5).astype(int)
    red, green, blue = make_dismph_colors()
    # make [m, n, 3] array by looking up each phase color
    rgb = np.stack((red[phase_idxs], green[phase_idxs], blue[phase_idxs]), axis=-1)

    # Now darken RGB values according to image magnitude
    if scalemag:
        mag = scale_mag(ifg, expval=expval, max_pct=max_pct)
        # mag /= np.max(mag)
        mag /= np.percentile(mag, max_pct)
        mag_scaling = mag[:, :, np.newaxis]  # Match rgb shape
    else:
        # Otherwise, just the phase Image
        mag_scaling = 1

    # breakpoint()
    img = Image.fromarray((mag_scaling * rgb).astype("uint8"))
    return img


def plot_ifg(
    ifg,
    phase_cmap="dismph",
    mag_cmap="gray",
    title="",
    expval=0.3,
    max_pct=99,
    subplot_layout=None,
    figsize=None,
    zero_mean_phase=False,
    log_amp=True,
    **kwargs,
):
    ifg = np.nan_to_num(ifg, copy=True, nan=0)
    if subplot_layout is None:
        # Decide to layout in rows or cols
        rowsize, colsize = figsize[::-1] if figsize is not None else ifg.shape
        subplot_layout = (1, 3) if colsize > rowsize else (3, 1)

    if figsize is None:
        figsize = (4 * subplot_layout[1], 4 * subplot_layout[0])
    fig, axes = plt.subplots(*subplot_layout, sharex=True, sharey=True, figsize=figsize)

    # mag = scale_mag(ifg)
    mag = np.abs(ifg)
    ax = axes[0]
    if log_amp:
        # Note: this PowerNorm does the same thing as my "scale_mag", but lets you see the actual values
        norm = mpl.colors.PowerNorm(gamma=expval, vmax=np.percentile(mag, 99.5))
    else:
        norm = None
    axim = ax.imshow(mag, norm=norm, cmap=mag_cmap)
    fig.colorbar(axim, ax=ax, extend="max")

    phase = phase_to_2pi(ifg)
    if zero_mean_phase:
        phase -= phase.mean()
    ax = axes[1]
    # Note: other interpolations (besides nearest/None) make dismph colorscheme look weird
    axim = ax.imshow(phase, cmap=phase_cmap, interpolation="nearest")
    fig.colorbar(axim, ax=ax)

    ax = axes[2]
    dismph_img = make_mph_image(ifg, expval=expval, max_pct=max_pct)
    axim = ax.imshow(dismph_img)

    if title:
        fig.suptitle(title)
        # axes[0].set_title(title)
    fig.tight_layout()
    return axes


def get_unique_cm(arr, name="jet"):
    """Get a colormap with unique colors for each value in arr."""
    return mpl.cm.get_cmap(name, len(np.unique(arr)))


def plot_cc(cc, ax, title="", cmap="jet"):
    """Plot a connected component image with unique colors."""
    axim = ax.imshow(cc, cmap=get_unique_cm(cc, name=cmap), interpolation="nearest")
    fig = ax.figure
    fig.colorbar(axim, ax=ax, ticks=np.arange(np.min(cc), np.max(cc) + 1))
    ax.set_title(title)


def plot_rewrapped(unw, ax=None, cmap="dismph", show_cbar=True, wrap_level=3 * np.pi):
    # Rewrap the unwrapped for easiest visual of differences
    if ax is None:
        fig, ax = plt.subplots()
    vmax, vmin = wrap_level, 0
    axim = ax.imshow(
        np.mod(unw, wrap_level),
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        interpolation="nearest",
    )
    if show_cbar:
        fig = ax.get_figure()
        fig.colorbar(axim, ax=ax)


def get_fig_ax(fig, ax, **figkwargs):
    """Handle passing None to either fig or ax by creating new, returns both"""
    if not fig and not ax:
        fig = plt.figure(**figkwargs)
        ax = fig.gca()
    elif ax and not fig:
        fig = ax.figure
    elif fig and not ax:
        ax = fig.gca()
    return fig, ax


def plot_image(
    img=None,
    filename=None,
    dset=None,
    fig=None,
    ax=None,
    cmap=DEFAULT_CMAP,
    rsc_data=None,
    title="",
    label="",
    xlabel="",
    ylabel="",
    vm=None,
    twoway=True,
    vmin=None,
    vmax=None,
    extent=None,
    bbox=None,
    aspect="auto",
    perform_shift=False,
    colorbar=True,
    **figkwargs,
):
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
        figkwargs: pass through to plt.figure() (e.g. figsize, sharex,...)
    Returns:
        fig, ax, axes_image
    """
    if filename and dset:
        from apertools import sario
        import h5py

        with h5py.File(filename, "r") as f:
            img = f[dset][:]
        rsc_data = sario.load_dem_from_h5(filename)

    nrows, ncols = img.shape
    if not extent:
        if bbox:
            extent = _bbox_to_extent(bbox)  # [bbox[0], bbox[2], bbox[1], bbox[3]]
        elif rsc_data:
            extent = latlon.grid_extent(**rsc_data)
        else:
            extent = (0, ncols, nrows, 0)

    fig, ax = get_fig_ax(fig, ax, **figkwargs)

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
    # plt.show(block=False)
    return fig, ax, axes_image


def _abs_max(img):
    """Find largest absolute value ignoring nans"""
    return np.nanmax(np.abs(img))


def _bbox_to_extent(bbox):
    return [bbox[0], bbox[2], bbox[1], bbox[3]]


def _get_vminmax(img, vm=None, vmin=None, vmax=None, twoway=True):
    vm = vm or np.nanmax(np.abs(img))
    if twoway:
        vmax = vm if vmax is None else vmax
        vmin = -vm if vmin is None else vmin
    else:
        vmax = vm if vmax is None else vmax
        vmin = 0 if vmin is None else vmin
    return vmin, vmax


def hvplot_stack(
    da,
    x="lon",
    y="lat",
    z="date",
    vmin=None,
    vmax=None,
    cmap="seismic_wide_y_r",
    twoway=True,
    vm=None,
    height=400,
    width=600,
):
    """Make a 2-panel interactive plot with clickable timeseries"""
    import panel as pn
    import holoviews as hv

    def get_timeseries(xc, yc):
        return da.sel(indexers={x: xc, y: yc}, method="nearest").hvplot(z)

    # pn.extension()
    vmin, vmax = _get_vminmax(da, vm=vm, vmin=vmin, vmax=vmax, twoway=twoway)

    image, select = da.hvplot(
        x,
        y,
        # widgets={z: pn.widgets.Select},
        widgets={z: pn.widgets.DiscreteSlider},
        cmap=cmap,
        clim=(vmin, vmax),
        height=height,
        width=width,
    )
    stream = hv.streams.Tap(source=image.object, x=da[x][0].item(), y=da[y][0].item())

    return pn.Column(
        image, select, pn.bind(get_timeseries, xc=stream.param.x, yc=stream.param.y)
    )


def view_stack(
    stack,
    display_img,
    slclist=None,
    label="Centimeters",
    cmap=DEFAULT_CMAP,
    perform_shift=False,
    title="",
    vmin=None,
    vmax=None,
    legend_loc="upper left",
    lat_lon=False,
    rsc_data=None,
    line_plot_kwargs=None,
    timeline_callback=None,
):
    """Displays an image from a stack, allows you to click for timeseries

    Args:
        stack (ndarray): 3D np.ndarray, 1st index is image number
            i.e. the idx image is stack[idx, :, :]
        slclist (list[datetime]): Optional: times of acquisition for
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
    if slclist is None:
        slclist = np.arange(stack.shape[0])
    try:
        slclist = [s.date() for s in slclist]
    except ValueError:
        pass
    slclist = np.array(slclist)

    imagefig = plt.figure()

    if isinstance(display_img, int):
        img = stack[display_img, :, :]
    # TODO: is this the best way to check if it's ndarray?
    elif np.ndim(display_img) == 2:
        img = display_img
    else:
        raise ValueError("display_img must be an int or ndarray-like obj")

    title = title or "Deformation Time Series"  # Default title
    _, _, axes_img = plot_image(
        img,
        fig=imagefig,
        title=title,
        cmap=cmap,
        label=label,
        vmin=vmin,
        vmax=vmax,
        perform_shift=perform_shift,
    )

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(bottom=0.25)
    # Make a horizontal slider to control the frequency.
    axdate = plt.axes([0.05, 0.1, 0.75, 0.03])
    date_slider = DateSlider(
        ax=axdate,
        label="Date",
        valmin=mdates.date2num(slclist[0]),
        valmax=mdates.date2num(slclist[-1]),
        valinit=mdates.date2num(slclist[-1]),
        valfmt="%Y-%m-%d",
    )

    def update_slider(val):
        # The function to be called anytime a slider's value changes
        val_dt = mdates.num2date(val).date()
        closest_idx = np.argmin(np.abs(slclist - val_dt))
        axes_img.set_data(stack[closest_idx])
        imagefig.canvas.draw()

    # register the update function with each slider
    date_slider.on_changed(update_slider)

    timefig = plt.figure()

    plt.title(title)
    legend_entries = []
    if not line_plot_kwargs:
        line_plot_kwargs = dict(
            marker="o", linestyle="dashed", linewidth=1, markersize=4
        )

    def onclick(event):
        # Ignore right/middle click, clicks off image
        if event.button != 1 or not event.inaxes:
            return
        # Check if the toolbar has zoom or pan active
        # https://stackoverflow.com/a/20712813
        # MPL version 3.3: https://stackoverflow.com/a/63447351
        if mpl.__version__ >= "3.3":
            state = imagefig.canvas.manager.toolbar.mode
            if state != "":  # Zoom/other tool is active
                return
        else:
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
            lat, lon = latlon.rowcol_to_latlon(row, col, rsc_data)
            legend_entries.append("Lat {:.3f}, Lon {:.3f}".format(lat, lon))
        else:
            legend_entries.append("Row %s, Col %s" % (row, col))

        fig = plt.figure(2)
        ax = fig.gca()
        ax.plot(slclist, timeline, **line_plot_kwargs)
        ax.legend(legend_entries, loc=legend_loc)
        x_axis_str = "SAR image date" if slclist is not None else "Image number"
        ax.set_xlabel(x_axis_str)
        timefig.autofmt_xdate()
        ax.set_ylabel(label)

        timefig.canvas.draw()

    imagefig.canvas.mpl_connect("button_press_event", onclick)
    plt.show(block=True)


from ipywidgets import interact, widgets
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime


def view_stack_improved(
    stack,
    display_img,
    slclist=None,
    label="Centimeters",
    cmap="seismic",
    perform_shift=False,
    title="",
    vmin=None,
    vmax=None,
    legend_loc="upper left",
    lat_lon=False,
    rsc_data=None,
    line_plot_kwargs=None,
    timeline_callback=None,
):
    """Displays an image from a stack, allows you to click for timeseries"""

    if slclist is None:
        slclist = np.arange(stack.shape[0])
    else:
        slclist = [s.date() for s in slclist]
    slclist = np.array(slclist)

    if isinstance(display_img, int):
        img = stack[display_img, :, :]
    elif np.ndim(display_img) == 2:
        img = display_img
    else:
        raise ValueError("display_img must be an int or ndarray-like obj")

    title = title or "Deformation Time Series"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    ax1.set_title(title)
    im = ax1.imshow(
        img,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
    )
    fig.colorbar(im, ax=ax1, label=label)

    # ipywidgets slider
    date_slider = widgets.SelectionSlider(
        options=[(date.strftime("%Y-%m-%d"), idx) for idx, date in enumerate(slclist)],
        description="Date:",
        orientation="horizontal",
        layout=widgets.Layout(width="80%"),
    )

    def update_slider(idx):
        im.set_data(stack[idx])
        fig.canvas.draw()

    interact(update_slider, idx=date_slider)

    def onclick(event):
        if event.button != 1 or not event.inaxes:
            return
        ax2.clear()
        row, col = int(event.ydata), int(event.xdata)
        if row >= img.shape[0] or col >= img.shape[1]:
            return
        timeline = stack[:, row, col]

        if timeline_callback is not None:
            timeline_callback(timeline, row, col)

        if lat_lon:
            lat, lon = latlon.rowcol_to_latlon(row, col, rsc_data)
            legend_label = "Lat {:.3f}, Lon {:.3f}".format(lat, lon)
        else:
            legend_label = "Row %s, Col %s" % (row, col)

        ax2.plot(slclist, timeline, **(line_plot_kwargs or {}))
        ax2.legend([legend_label], loc=legend_loc)
        x_axis_str = "SAR image date" if slclist is not None else "Image number"
        ax2.set_xlabel(x_axis_str)
        ax2.set_ylabel(label)

    fig.canvas.mpl_connect("button_press_event", onclick)


def equalize_and_mask(image, low=1e-6, high=2, fill_value=np.inf, db=True):
    """Clips an image to increase contrast"""
    # Mask the invalids, then mask zeros, then clip rest
    im = np.clip(utils.mask_zeros(np.ma.masked_invalid(image)), low, high)
    if fill_value:
        im.set_fill_value(fill_value)
    return utils.db(im) if db else im


def animate_stack(
    stack,
    pause_time=200,
    display=True,
    titles=None,
    label=None,
    save_title=None,
    cmap_name="seismic",
    shifted=True,
    vmin=None,
    vmax=None,
    **savekwargs,
):
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
        titles = ["" for _ in range(num_images)]  # blank titles, same length
    if np.iscomplexobj(stack):
        stack = np.abs(stack)

    # Use the same stack min and stack max (or vmin/vmax) for all colorbars/ color ranges
    vmin = np.min(stack) if vmin is None else vmin
    vmax = np.max(stack) if vmax is None else vmax
    cmap = (
        cmap_name
        if not shifted
        else make_shifted_cmap(vmin=vmin, vmax=vmax, cmap_name=cmap_name)
    )

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
        return (axes_image,)

    stack_ani = animation.FuncAnimation(
        fig,
        update_im,
        frames=range(num_images),
        interval=pause_time,
        blit=False,
        repeat=True,
    )

    if save_title:
        logger.info("Saving to %s", save_title)
        stack_ani.save(save_title, writer="imagemagick", **savekwargs)

    if display:
        plt.show()
    return stack_ani


def make_figure_noborder():
    fig = plt.figure(frameon=False)

    # To make the content fill the whole figure
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax


def set_aspect_image(fig, img, height=4):
    """Adjusts sizes to match image data ratio

    Can pick a height (default 4), and keeps data rows/col ratio
    """
    nrows, ncols = img.shape
    width = ncols / nrows * height
    print("width", width, "height", height)
    fig.set_size_inches(width, height)


def save_paper_figure(fig, fname, axis_off=True, width=None, dpi=300):
    fig.tight_layout()
    if axis_off:
        plt.axis("off")
    if width:
        fig.set_size_inches(get_figsize(width))
    print("Saving %s" % fname)
    fig.savefig(fname, bbox_inches="tight", transparent=True, dpi=dpi)


def get_figsize(width="half", fraction=1, ratio=0.618, subplots=(1, 1)):
    """Get figure dimensions to avoid scaling in LaTeX.

    Args:
    width (float): Document textwidth or columnwidth in pts
    fraction (float): optional. Fraction of the width which you wish the figure to occupy

    Returns:
        fig_dim: (tuple) Dimensions of figure in inches

    Notes:
        for the IEEE TGRS article:
        \showthe\columnwidth -> 252.0pt.
        \showthe\textwidth -> 516.0pt.
    Default ratio is Golden ratio to set aesthetic figure height
        https://disq.us/p/2940ij3
    Source:
        https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    """
    if width == "half":
        width = 252.0
    elif width == "full":
        width = 516.0
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def plot_shapefile(filename, fig=None, ax=None, z=None):
    # Credit: https://gis.stackexchange.com/a/152331
    import shapefile

    fig, ax = get_fig_ax(fig, ax)

    with shapefile.Reader(filename) as sf:
        for shape in sf.shapeRecords():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            if z is not None:
                ax.plot3D(x, y, z, "b")
            else:
                ax.plot(x, y, "b")

        plt.show()


# def plotcompare(fnames, dset="velos", vmax=25, vmin=-25, cmap="seismic_wide", **kwargs):
def plot_img_diff(
    arrays=None,
    dset="defo_lowess",
    fnames=[],
    vm=6,
    vmax=None,
    vmin=None,
    twoway=True,
    titles=[],
    show_diff=True,
    vdiff=1,
    diff_title="left - middle",
    cmap=DEFAULT_CMAP,
    axes=None,
    axis_off=False,
    cbar_label="",
    show=True,
    figsize=None,
    interpolation=None,
    aspect=None,
    bbox=None,
    extent=None,
    share=True,
    use_proplot=True,
    imshow_kwargs={},
    **kwargs,
):
    """Plot two images for comparison, (and their difference if `show_diff`)"""
    # import proplot as pplt

    if arrays is None:
        from apertools import sario

        arrays = [sario.load(f, dset=dset, **kwargs) for f in fnames]

    n = len(arrays)
    ncols = n + 1 if show_diff else n
    vmin, vmax = _get_vminmax(arrays[0], vm=vm, vmin=vmin, vmax=vmax, twoway=twoway)
    # print(f"{vmin} {vmax}")
    if axes is None:
        fig, axes = plt.subplots(
            # fig, axes = pplt.subplots(
            ncols=ncols,
            sharex=share,
            sharey=share,
            figsize=figsize,
        )
    else:
        try:
            axes = axes.ravel()
        except AttributeError:
            pass
        fig = axes[0].figure

    for ii in range(n):
        if bbox:
            extent = [bbox[0], bbox[2], bbox[1], bbox[3]]

        ax = axes[ii]
        axim = ax.imshow(
            arrays[ii],
            cmap=cmap,
            vmax=vmax,
            vmin=vmin,
            interpolation=interpolation,
            extent=extent,
            **imshow_kwargs,
        )
        if titles:
            ax.set_title(titles[ii])
        # numbers: weird
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        cbar = fig.colorbar(axim, ax=ax, fraction=0.033, pad=0.04)
        # # Proplot version:
        # ax.colorbar(axim, loc="r", label=cbar_label)
        if axis_off:
            ax.set_axis_off()
        if aspect:
            ax.set_aspect(aspect)
    # fig.colorbar(axim, ax=axes[n - 1])
    if show_diff:
        # Now different image at end
        diff_arr = arrays[0] - arrays[1]
        # the diff is always two way, even if arrays are positive only
        vmin, vmax = _get_vminmax(diff_arr, vm=vdiff, twoway=True)
        ax = axes[-1]
        axim = ax.imshow(
            diff_arr,
            cmap=cmap,
            vmax=vmin,
            vmin=vmax,
            interpolation=interpolation,
            extent=extent,
            **imshow_kwargs,
        )
        ax.set_title(diff_title)
        if axis_off:
            ax.set_axis_off()
        if aspect:
            ax.set_aspect(aspect)
        cbar = fig.colorbar(axim, ax=ax, fraction=0.033, pad=0.04)
        cbar.set_label(cbar_label)
    else:
        cbar.set_label(cbar_label)
        # ax.colorbar(axim, loc="r", label=cbar_label)
    # [f.close() for f in files]
    if show:
        plt.show(block=False)
    return fig, axes


def rescale_and_color(in_name, outname, vmin=None, vmax=None, cmap=None):
    import rasterio as rio

    with rio.open(in_name) as src:
        arr = src.read(1)
        mask = arr == src.nodata
        meta = src.meta
    out_dt = "uint8"
    meta["dtype"] = out_dt

    if vmax is None or vmin is None:
        arr_valid = arr[~mask]
        vmin, vmax = np.min(arr_valid), np.max(arr_valid)

    arr = np.clip(arr, vmin, vmax)  # range: [-vmin, vmax]
    arr = (vmin + arr) / (vmax - vmin)  # [-1, 1]
    arr = 1 + 255 * (1 + arr)  # [1, 255]
    arr = np.clip(arr, 1, 255).astype(out_dt)
    arr[mask] = 0  # reset nodata
    with rio.open(outname, "w", **meta) as dst:
        dst.write(arr.astype(out_dt), 1)

        if cmap:
            dst.write_colormap(1, cmap_to_dict(cmap))
    return arr


save_as_rgv_tiff = rescale_and_color


def create_marker_from_svg(
    filename, rotation=0, shift=True, flip=True, scale=1, **kwargs
):
    """Create a marker from an SVG file

    Based on https://petercbsmith.github.io/marker-tutorial.html
    Requires svgpathtools and svgpath2mpl

    Example:
        >>> marker = create_marker_from_svg("/path/to/marker.svg", rotation=45)
        >>> ax.plot(x, y, marker=marker)
    """
    import matplotlib as mpl
    from svgpathtools import svg2paths
    from svgpath2mpl import parse_path

    path, attributes = svg2paths(filename)
    marker = parse_path(attributes[0]["d"])
    if shift:
        marker.vertices -= marker.vertices.mean(axis=0)
    if flip:
        marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
        marker = marker.transformed(mpl.transforms.Affine2D().scale(-1, 1))
    if rotation:
        marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(rotation))
    return marker


def cmap_to_dict(cmap_name, vmin=None, vmax=None):
    # for matplotlib.colors.LinearSegmentedColormap
    # syd = {r: tuple(256*np.array(sy(r))) for r in range(256) }
    # cmy(.4, bytes=True)
    # (219, 237, 200, 255)
    cmap = mpl.cm.get_cmap(cmap_name)
    if vmin is None or vmax is None:
        vmin, vmax = 0, 1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    vals = np.linspace(vmin, vmax, cmap.N)
    color_tuple_dict = {idx: cmap(norm(v), bytes=True) for idx, v in enumerate(vals)}
    # {1: (219, 237, 200, 255),... }
    return color_tuple_dict


def map_background(
    *,
    bbox=None,
    image=None,
    pad_pct=0.3,
    bbox_image=None,
    zoom_level=8,
    fig=None,
    ax=None,
    coastlines=False,
    show_ticks=True,
    tickside="left",
    img_zorder=2,
    figsize=None,
    crs_name="PlateCarree",
    img_epsg=None,
    **imshow_kwargs,
):
    """Plot the raster `img` on top of background tiles
    Inputs:
        img (ndarray): raster image
        bbox (tuple[float]): (left, bottom, right, top)
        fig (matplotlib.figure): optional, existing figure to use
    """
    import cartopy.crs as ccrs
    from cartopy.io import img_tiles

    # tiler = img_tiles.Stamen("terrain-background")
    # tiler = img_tiles.GoogleTiles(style="satellite")
    mykey = "pk.eyJ1Ijoic2NvdHRzdGFuaWUiLCJhIjoiY2s3Nno3bmE5MDJlbDNmcGNpanV0ZzJ3MCJ9.PyaQ_iwKFcFcRr-EveCObA"
    # https://github.com/SciTools/cartopy/issues/1965#issuecomment-992603403
    # tiler = img_tiles.MapboxTiles(map_id="satellite-v9", access_token=mykey)
    tiler = img_tiles.GoogleTiles(style="satellite")
    crs = tiler.crs

    # if fig is None and ax is None:
    # print('ADDED FIG')
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=crs)
    else:
        fig = ax.figure

    if bbox is None:
        try:
            bbox = _get_rio_bbox(image)
        except AttributeError:
            bbox = (0, 0, 1, 1)
    # matplotlib wants extent different than gdal/rasterio convention
    pad_pct = pad_pct or 0.0
    extent = padded_extent(bbox, pad_pct)
    try:
        import rasterio as rio
        from cartopy.crs import Projection

        with rio.Env(OSR_WKT_FORMAT="WKT2_2018"):
            if img_epsg is not None:
                rio_crs = rio.crs.CRS.from_epsg(img_epsg)
            else:
                rio_crs = image.rio.crs
            crs = Projection(rio_crs)
    except AttributeError:
        crs = getattr(ccrs, crs_name)()
    ax.set_extent(extent, crs=crs)

    ax.add_image(tiler, zoom_level)
    if image is not None:
        if bbox_image is None:
            bbox_image = _get_rio_bbox(image)
        extent_img = padded_extent(bbox_image, 0.0)
        axim = ax.imshow(
            image,
            transform=crs,
            extent=extent_img,
            origin="upper",
            zorder=img_zorder,
            **imshow_kwargs,
        )

    if show_ticks:
        add_ticks(ax, side=tickside)

    if coastlines:
        ax.coastlines("10m")
    return fig, ax


def plot_image_with_background(
    image, figsize=None, cbar_label=None, tile_zoom_level=9, **imshow_kwargs
):
    import cartopy.crs as ccrs
    from cartopy.io import img_tiles
    # Read the raster image and get its extent

    # Create a figure and add a GeoAxes with a projection
    # import proplot as pplt
    # fig = pplt.figure(figsize=figsize)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add the satellite imagery from Google Tiles
    tiler = img_tiles.GoogleTiles(style="satellite")
    # tiler = img_tiles.Stamen(style="terrain")
    ax.add_image(tiler, tile_zoom_level, interpolation="bicubic")

    extent = padded_extent(image.rio.bounds(), 0.0)
    print(extent)
    # Plot the raster image on top of the satellite background
    axim = ax.imshow(
        image,
        origin="upper",
        extent=extent,
        transform=ccrs.PlateCarree(),
        zorder=2,
        **imshow_kwargs,
    )
    cbar = fig.colorbar(axim)
    cbar.set_label(cbar_label)

    # Set the extent for the GeoAxes to the raster image's extent
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    return fig, ax


def padded_extent(bbox, pad_pct):
    """Return a padded extent, given a bbox and a percentage of padding"""
    left, bot, right, top = bbox
    padx = pad_pct * (right - left) / 2
    pady = pad_pct * (top - bot) / 2
    return (left - padx, right + padx, bot - pady, top + pady)


def add_ticks(ax, side="right", resolution: float = 1):
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    left, right, bot, top = ax.get_extent(ccrs.PlateCarree())
    # print(left, bot, right, top)
    # lon_ticks = np.arange(np.ceil(left), np.floor(right), step=resolution)
    # lat_ticks = np.arange(np.ceil(bot), np.floor(top), step=resolution)
    bounds = (left, bot, right, top)
    lon_ticks, lat_ticks = generate_ticks(bounds, resolution=resolution)
    print(lon_ticks, lat_ticks)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.yaxis.tick_left()
    # print("added ticks")
    ax.xaxis.tick_bottom()


def generate_ticks(bounds, resolution, offset=0):
    """
    Generate xticks and yticks for a raster image based on given bounds, resolution, and offset.

    Parameters
    ----------
    bounds : tuple of float
        The bounds of the raster image in the form (left, bottom, right, top).
    resolution : float
        The spacing/rounding resolution for the ticks.
    offset : float, optional
        The offset to be added to the tick positions, by default 0.

    Returns
    -------
    xticks : numpy.ndarray
        The generated xticks adjusted to the specified resolution and offset.
    yticks : numpy.ndarray
        The generated yticks adjusted to the specified resolution and offset.

    Examples
    --------
    >>> bounds = (10, 20, 50, 70)
    >>> resolution = 5
    >>> generate_ticks(bounds, resolution)
    (array([10, 15, 20, 25, 30, 35, 40, 45, 50]),
     array([20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]))

    >>> generate_ticks(bounds, resolution, offset=2)
    (array([12, 17, 22, 27, 32, 37, 42, 47]),
     array([22, 27, 32, 37, 42, 47, 52, 57, 62]))
    """

    def _snap_bounds_to_res(bounds, resolution):
        left, bottom, right, top = bounds
        # Adjust the extents
        new_left = np.ceil(left / resolution) * resolution
        new_right = np.floor(right / resolution) * resolution
        new_bottom = np.ceil(bottom / resolution) * resolution
        new_top = np.floor(top / resolution) * resolution

        return [new_left, new_bottom, new_right, new_top]

    left, bottom, right, top = _snap_bounds_to_res(bounds, resolution)

    # Generate xticks from left to right bounds
    xticks = np.arange(left, right + resolution, resolution) + offset
    # Filter xticks to be within the bounds
    xticks = xticks[(xticks >= left) & (xticks <= right)]

    # Generate yticks from bottom to top bounds
    yticks = np.arange(bottom, top + resolution, resolution) + offset
    # Filter yticks to be within the bounds
    yticks = yticks[(yticks >= bottom) & (yticks <= top)]

    return xticks, yticks


def _make_line_collections(ax, *, ticks=None, loc="bottom", lw=2, **kwargs):
    import itertools
    from matplotlib.collections import LineCollection

    left, right, bot, top = ax.get_extent()
    if ticks is None:
        if loc in ["top", "bottom", "bot"]:
            ticks = [left, *ax.get_xticks(), right]
            print(ticks)
            ticks = np.unique(np.array(ticks).round(2))
            # ticks = np.array(
            #     [t.get_position() for t in ax.get_xmajorticklabels()]
            # ).round(0)
        else:
            ticks = [bot, *ax.get_yticks(), top]
            print(ticks)
            ticks = np.unique(np.array(ticks).round(2))
            # ticks = np.array(
            #     [t.get_position() for t in ax.get_ymajorticklabels()]
            # ).round(0)
    print(f"{loc = }, ticks: {ticks}")
    if loc in ("bottom", "bot"):
        points = np.array([ticks, bot * np.ones_like(ticks)]).T.reshape(-1, 1, 2)
    elif loc == "top":
        points = np.array([ticks, top * np.ones_like(ticks)]).T.reshape(-1, 1, 2)
    elif loc == "left":
        points = np.array([left * np.ones_like(ticks), ticks]).T.reshape(-1, 1, 2)
    elif loc == "right":
        points = np.array([right * np.ones_like(ticks), ticks]).T.reshape(-1, 1, 2)
    # print(points)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc_outline = LineCollection(
        segments,
        colors="black",
        linewidth=lw + 1,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
    bws = itertools.cycle(["k", "white"])
    colors = [next(bws) for _ in range(len(ticks) - 1)]
    print(f"""colors: {colors}""")
    lc = LineCollection(
        segments,
        colors=colors,
        linewidth=lw,
        transform=ax.get_xaxis_transform(),
        clip_on=False,
    )
    return lc_outline, lc


def add_zebra_frame(ax, lw=2, crs="pcarree", zorder=None):
    import itertools
    import matplotlib.patheffects as pe

    ax.spines["geo"].set_visible(False)
    left, right, bot, top = ax.get_extent()
    bws = itertools.cycle(["k", "white"])

    xticks = sorted([left, *ax.get_xticks(), right])
    xticks = np.unique(np.array(xticks))
    yticks = sorted([bot, *ax.get_yticks(), top])
    yticks = np.unique(np.array(yticks))
    for ticks, which in zip([xticks, yticks], ["lon", "lat"]):
        for idx, (start, end) in enumerate(zip(ticks, ticks[1:])):
            bw = next(bws)
            if which == "lon":
                xs = [[start, end], [start, end]]
                ys = [[bot, bot], [top, top]]
            else:
                xs = [[left, left], [right, right]]
                ys = [[start, end], [start, end]]

            # For first and lastlines, used the "projecting" effect
            capstyle = "butt" if idx not in (0, len(ticks) - 2) else "projecting"
            for xx, yy in zip(xs, ys):
                ax.plot(
                    xx,
                    yy,
                    color=bw,
                    linewidth=lw,
                    clip_on=False,
                    transform=crs,
                    zorder=zorder,
                    solid_capstyle=capstyle,
                    path_effects=[
                        pe.Stroke(linewidth=lw + 1, foreground="black"),
                        pe.Normal(),
                    ],
                )


import itertools
from matplotlib.patheffects import Stroke, Normal
import numpy as np
import cartopy.mpl.geoaxes
import cartopy.crs as ccrs


def zebra_frame(self, lw=3, crs=None, zorder=None, iFlag_outer_frame_in=None):
    # Alternate black and white line segments
    print("???????")
    bws = itertools.cycle(["k", "w"])
    self.spines["geo"].set_visible(False)

    if iFlag_outer_frame_in is not None:
        # get the map spatial reference
        left, right, bottom, top = self.get_extent()
        crs_map = self.projection
        xticks = np.arange(left, right + (right - left) / 9, (right - left) / 8)
        yticks = np.arange(bottom, top + (top - bottom) / 9, (top - bottom) / 8)
        # check spatial reference are the same
        print(xticks, yticks)
        pass
    else:
        crs_map = crs
        xticks = sorted([*self.get_xticks()])
        xticks = np.unique(np.array(xticks))
        yticks = sorted([*self.get_yticks()])
        yticks = np.unique(np.array(yticks))

    print(xticks, yticks, crs_map)
    for ticks, which in zip([xticks, yticks], ["lon", "lat"]):
        for idx, (start, end) in enumerate(zip(ticks, ticks[1:])):
            bw = next(bws)
            if which == "lon":
                xs = [[start, end], [start, end]]
                ys = [[yticks[0], yticks[0]], [yticks[-1], yticks[-1]]]
            else:
                xs = [[xticks[0], xticks[0]], [xticks[-1], xticks[-1]]]
                ys = [[start, end], [start, end]]

            # For first and last lines, used the "projecting" effect
            capstyle = "butt" if idx not in (0, len(ticks) - 2) else "projecting"
            for xx, yy in zip(xs, ys):
                self.plot(
                    xx,
                    yy,
                    color=bw,
                    linewidth=max(0, lw - self.spines["geo"].get_linewidth() * 2),
                    clip_on=False,
                    transform=crs_map or ccrs.PlateCarree(),
                    zorder=zorder,
                    solid_capstyle=capstyle,
                    # Add a black border to accentuate white segments
                    path_effects=[
                        Stroke(linewidth=lw, foreground="black"),
                        Normal(),
                    ],
                )


setattr(cartopy.mpl.geoaxes.GeoAxes, "zebra_frame", zebra_frame)


def get_data_offset(ax, dx_pix=0.01, dy_pix=0.01, lw=5):
    if lw:
        dx_pix = lw / 72 / 2
        dy_pix = lw / 72 / 2
    FC_to_DC = ax.transData.inverted().transform
    NDC_to_FC = ax.transAxes.transform
    NDC_to_DC = lambda x: FC_to_DC(NDC_to_FC(x))

    dy = NDC_to_DC([0.00, dy_pix]) - NDC_to_DC([0.00, 0.00])
    dx = NDC_to_DC([dx_pix, 0.0]) - NDC_to_DC([0.00, 0.00])
    return dx[0], dy[1]


def _get_rio_bbox(img):
    try:
        if "lon" in img.dims:
            bbox = (img.lon.min(), img.lat.min(), img.lon.max(), img.lat.max())
            bbox = [b.item() for b in bbox]
        elif "x" in img.dims:
            bbox = img.rio.bounds()
    except:
        raise ValueError("bbox must be provided if `img` is not an xarray DataArray")
    return bbox


def plot_rect(
    ax,
    lon=None,
    lat=None,
    w=0.05,
    bbox=None,
    edgecolor="k",
    lw=3,
    zorder=5,
):
    from matplotlib.patches import Rectangle

    if lon is not None and lat is not None and w is not None:
        xy = (lon - w / 2, lat - w / 2)
        width = height = w
    elif bbox is not None:
        left, bot, right, top = bbox
        xy = (left, bot)
        width = right - left
        height = top - bot
    rect = Rectangle(
        xy, width, height, facecolor="none", edgecolor=edgecolor, lw=lw, zorder=zorder
    )
    return ax.add_patch(rect)


def map_img(
    image=None,
    bbox=None,
    pad_pct=0.0,
    ax=None,
    crs=None,
    add_colorbar=True,
    **imshow_kwargs,
):
    import cartopy.crs as ccrs

    if crs is None:
        crs = ccrs.PlateCarree()

    if ax is None:
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(1, 1, 1, projection=crs)

    if bbox is None:
        bbox = _get_rio_bbox(image)

    extent_img = padded_extent(bbox, 0.0)
    axim = ax.imshow(
        image, transform=crs, extent=extent_img, origin="upper", **imshow_kwargs
    )
    # if add_colorbar:
    # ax.colorbar(axim, loc='r')
    extent = padded_extent(bbox, pad_pct)
    ax.set_extent(extent, crs=crs)
    return ax, axim


def scale_bar0(
    ax,
    location,
    length,
    metres_per_unit=1000,
    unit_name="km",
    tol=0.01,
    angle=0,
    color="black",
    linewidth=3,
    text_offset=0.005,
    ha="center",
    va="bottom",
    plot_kwargs=None,
    text_kwargs=None,
    **kwargs,
):
    """Add a scale bar to CartoPy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Args:
        ax:              CartoPy axes.
        location:        Position of left-side of bar in axes coordinates.
        length:          Geodesic length of the scale bar.
        metres_per_unit: Number of metres in the given unit. Default: 1000
        unit_name:       Name of the given unit. Default: 'km'
        tol:             Allowed relative error in length of bar. Default: 0.01
        angle:           Anti-clockwise rotation of the bar.
        color:           Color of the bar and text. Default: 'black'
        linewidth:       Same argument as for plot.
        text_offset:     Perpendicular offset for text in axes coordinates.
                         Default: 0.005
        ha:              Horizontal alignment. Default: 'center'
        va:              Vertical alignment. Default: 'bottom'
        **plot_kwargs:   Keyword arguments for plot, overridden by **kwargs.
        **text_kwargs:   Keyword arguments for text, overridden by **kwargs.
        **kwargs:        Keyword arguments for both plot and text.
    """
    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {"linewidth": linewidth, "color": color, **plot_kwargs, **kwargs}
    text_kwargs = {
        "ha": ha,
        "va": va,
        "rotation": angle,
        "color": color,
        **text_kwargs,
        **kwargs,
    }

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad, tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(
        *text_location,
        f"{length} {unit_name}",
        rotation_mode="anchor",
        transform=ax.transAxes,
        **text_kwargs,
    )


def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    import cartopy.crs as ccrs

    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        direction  Nonzero (2, 1)-shaped array, a direction vector.
        distance:  Positive distance to go past.
        dist_func: A two-argument function which returns distance.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment  from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        end:       Outer bound on point's location.
        distance:  Positive distance to travel.
        dist_func: Two-argument function which returns distance.
        tol:       Relative error in distance to allow.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(
            f"End is closer to start ({initial_distance}) than "
            f"given distance ({distance})."
        )

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Args:
        ax:       CartoPy axes.
        start:    Starting point for the line in axes coordinates.
        distance: Positive physical distance to travel.
        angle:    Anti-clockwise angle for the bar, in radians. Default: 0
        tol:      Relative error in distance to allow. Default: 0.01

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    import cartopy.geodesic as cgeo

    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cgeo.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        return geodesic.inverse(a_phys, b_phys).base[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(
    ax,
    length,
    location=(0.5, 0.05),
    scalewidth=3,
    units="km",
    m_per_unit=1000,
    zorder=1,
    lw=5,
    proj=None,
    utm_zone=None,
    utm_extent=None,
):
    """
    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    proj is the projection the axes are in
    length is the length of the scalebar in km.
    scalewidth is the thickness of the scalebar.
    units is the name of the unit
    m_per_unit is the number of meters in a unit


    Example:
    scale_len = 100 # km
    loc = (0.2, 0.03)
    scale_bar(ax, ax.projection, scale_len, location=loc, zorder=4)

    """
    import cartopy.crs as ccrs
    from matplotlib import patheffects

    if utm_zone is None or utm_extent is None:
        # find lat/lon center to find best UTM zone
        x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
        # Projection in metres
        utm = ccrs.UTM(
            latlon.utm_zone_from_lon((x0 + x1) / 2), southern_hemisphere=(y0 < 0)
        )
        # Get the extent of the plotted area in coordinates in metres
        x0, x1, y0, y1 = ax.get_extent(utm)
    else:
        if isinstance(utm_zone, int):
            utm = ccrs.UTM(utm_zone)
        else:
            utm = ccrs.CRS(utm_zone)
        x0, x1, y0, y1 = utm_extent

    print(x0, x1, y0, y1)
    # Turn the specified scalebar location into coordinates in metres
    sbcx = x0 + (x1 - x0) * location[0]
    sbcy = y0 + (y1 - y0) * location[1]
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * m_per_unit / 2, sbcx + length * m_per_unit / 2]
    # buffer for scalebar
    buffer = [patheffects.withStroke(linewidth=lw, foreground="w")]
    # Plot the scalebar with buffer
    # ax.plot(
    #     bar_xs,
    #     [sbcy, sbcy],
    #     transform=utm,
    #     color="k",
    #     linewidth=scalewidth,
    #     path_effects=buffer,
    #     zorder=zorder,
    # )
    # buffer for text
    buffer = [patheffects.withStroke(linewidth=lw // 2 + 1, foreground="w")]
    # Plot the scalebar label
    print(sbcx, sbcy)
    t0 = ax.text(
        sbcx,
        sbcy,
        str(length) + " " + units,
        transform=utm,
        horizontalalignment="center",
        verticalalignment="bottom",
        path_effects=buffer,
        zorder=zorder + 1,
    )
    # left = x0 + (x1 - x0) * 0.05
    # Plot the N arrow
    #     t1 = ax.text(left, sbcy, u'\u25B2\nN', transform=utm,
    #         horizontalalignment='center', verticalalignment='bottom',
    #         path_effects=buffer, zorder=2)
    # Plot the scalebar without buffer, in case covered by text buffer
    ax.plot(
        bar_xs,
        [sbcy, sbcy],
        transform=utm,
        color="k",
        linewidth=scalewidth,
        zorder=zorder + 2,
        solid_capstyle="butt",
    )


def plot_2d_arrays(
    arrays: list[np.ndarray],
    titles: list[str] | None = None,
    cmaps: list[str] | None = None,
    vmins: list[float] | None = None,
    vmaxes: list[float] | None = None,
    figsize: tuple[float, float] = (12, 12),
    imshow_kwargs: list[dict] | None = None,
) -> None:
    """Plot a list of 2D arrays in a grid of subplots with shared axes and colorbars.

    Parameters
    ----------
    arrays : List[np.ndarray]
        List of 2D numpy arrays to plot.
    titles : Optional[List[str]], optional
        List of titles for each subplot. If None, no titles are displayed.
    cmaps : Optional[List[str]], optional
        List of colormaps for each subplot. If None, 'viridis' is used for all.
    vmins : Optional[List[float]], optional
        List of minimum values for color scaling. If None, each array's minimum is used.
    vmaxes : Optional[List[float]], optional
        List of maximum values for color scaling. If None, each array's maximum is used.
    figsize : Tuple[float, float], optional
        Base figure size (width, height) in inches. The actual size may be adjusted.

    Returns
    -------
    None
        This function displays the plot but does not return any value.

    Notes
    -----
    This function creates a grid of subplots, each displaying one of the input 2D arrays.
    The grid is made as square as possible, adjusting for the number of arrays provided.
    Each subplot includes a colorbar and an optional title.
    """
    n = len(arrays)

    # Calculate grid dimensions
    nrows = ncols = ceil(sqrt(n))

    # Adjust figsize based on the number of plots
    aspect_ratio = figsize[0] / figsize[1]
    adjusted_figsize = (
        figsize[0] * ncols / sqrt(n),
        figsize[1] * nrows / (sqrt(n) * aspect_ratio),
    )

    # Create figure and axes
    fig, axes = plt.subplots(
        nrows, ncols, figsize=adjusted_figsize, sharex=True, sharey=True, squeeze=False
    )

    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    if isinstance(cmaps, (str, Colormap)):
        cmaps = [cmaps] * len(arrays)
    if isinstance(vmaxes, (float, int)):
        vmaxes = [vmaxes] * len(arrays)
    if isinstance(vmins, (float, int)):
        vmins = [vmins] * len(arrays)

    # Plot each array
    for i, arr in enumerate(arrays):
        # Get optional arguments for this subplot
        title = titles[i] if titles and i < len(titles) else None
        cmap = cmaps[i] if cmaps and i < len(cmaps) else "viridis"
        vmin = vmins[i] if vmins and i < len(vmins) else None
        vmax = vmaxes[i] if vmaxes and i < len(vmaxes) else None
        imshow_kw = imshow_kwargs[i] if imshow_kwargs is not None else {}

        # Create the plot
        im = axes[i].imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kw)

        # Add colorbar
        fig.colorbar(im, ax=axes[i])

        # Set title
        if title:
            axes[i].set_title(title)

    # Remove extra subplots
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and display
    fig.tight_layout()
    return fig, axes
