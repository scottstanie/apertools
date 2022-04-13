"""plotting.py: functions for visualizing insar products
"""
from __future__ import division, print_function
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


def set_style(size=15, nolatex=True, grid_on=False, cmap="viridis", weight="bold", minor_ticks=False):
    style = ["science", "no-latex"] if nolatex else "science"
    plt.style.use(style)
    style_dict = get_style(size, grid_on=grid_on, cmap=cmap, weight=weight, minor_ticks=minor_ticks)
    plt.rcParams.update(style_dict)
    try:
        import xarray as xr

        xr.set_options(cmap_divergent="RdBu")
    except:
        pass


def scale_mag(img, expval=0.3, max_pct=99.95):
    """Scale the magnitude of complex radar iamge for better display"""
    out = np.abs(img) ** expval
    max_val = np.percentile(out, max_pct)
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
            extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
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
    **kwargs,
):
    """Plot two images for comparison, (and their difference if `show_diff`)"""
    import proplot as pplt
    if arrays is None:
        from apertools import sario

        arrays = [sario.load(f, dset=dset, **kwargs) for f in fnames]

    n = len(arrays)
    ncols = n + 1 if show_diff else n
    vmin, vmax = _get_vminmax(arrays[0], vm=vm, vmin=vmin, vmax=vmax, twoway=twoway)
    # print(f"{vmin} {vmax}")
    if axes is None:
        # fig, axes = plt.subplots(
        fig, axes = pplt.subplots(
            ncols=ncols, sharex=share, sharey=share, figsize=figsize,
        )
    else:
        fig = axes.figure
    # axes = axes.ravel()

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
        )
        if titles:
            ax.set_title(titles[ii])
        # numbers: weird
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        # cbar = fig.colorbar(axim, ax=ax, fraction=0.033, pad=0.04)
        # cbar.set_label(cbar_label)
        # Proplot version:
        ax.colorbar(axim, loc='r', label=cbar_label)
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
        )
        ax.set_title("left - middle")
        if axis_off:
            ax.set_axis_off()
        if aspect:
            ax.set_aspect(aspect)
        # cbar = fig.colorbar(axim, ax=ax, fraction=0.033, pad=0.04)
        # cbar.set_label(cbar_label)
        ax.colorbar(axim, loc='r', label=cbar_label)
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


def create_marker_from_svg(filename, rotation=0, shift=True, flip=True, scale=1, **kwargs):
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
    marker = parse_path(attributes[0]['d'])
    if shift:
        marker.vertices -= marker.vertices.mean(axis=0)
    if flip:
        marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
        marker = marker.transformed(mpl.transforms.Affine2D().scale(-1,1))
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
    bbox,
    pad_pct=0.3,
    image=None,
    bbox_image=None,
    zoom_level=9,
    fig=None,
    ax=None,
    coastlines=False,
    show_ticks=True,
    tickside="left",
    **imshow_kwargs,
):
    """Plot the raster `img` on top of background tiles
    Inputs:
        img (ndarray): raster iamge
        bbox (tuple[float]): (left, bottom, right, top)
        fig (matplotlib.figure): optional, existing figure to use
    """
    import cartopy.crs as ccrs
    from cartopy.io import img_tiles

    tiler = img_tiles.Stamen("terrain-background")
    tiler = img_tiles.GoogleTiles(style="satellite")
    mykey = "pk.eyJ1Ijoic2NvdHRzdGFuaWUiLCJhIjoiY2s3Nno3bmE5MDJlbDNmcGNpanV0ZzJ3MCJ9.PyaQ_iwKFcFcRr-EveCObA"
    tiler = img_tiles.MapboxTiles(mykey, "satellite")
    crs = tiler.crs

    # if fig is None and ax is None:
    # print('ADDED FIG')
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=crs)
    else:
        fig = ax.figure

    # matplotlib wants extent different than gdal/rasterio convention
    pad_pct = pad_pct or 0.0
    extent = _padded_extent(bbox, pad_pct)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_image(tiler, zoom_level)
    if image is not None:
        if bbox_image is None:
            bbox_image = _get_rio_bbox(image)
        extent_img = _padded_extent(bbox_image, 0.0)
        ax.imshow(
            image,
            transform=ccrs.PlateCarree(),
            extent=extent_img,
            origin="upper",
            zorder=3,
            **imshow_kwargs,
        )

    if show_ticks:
        add_ticks(ax, side=tickside)

    if coastlines:
        ax.coastlines("10m")
    return fig, ax


def _padded_extent(bbox, pad_pct):
    """Return a padded extent, given a bbox and a percentage of padding"""
    left, bot, right, top = bbox
    padx = pad_pct * (right - left) / 2
    pady = pad_pct * (top - bot) / 2
    return (left - padx, right + padx, bot - pady, top + pady)



def add_ticks(ax, side="right"):
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    left, right, bot, top = ax.get_extent(ccrs.PlateCarree())
    lon_ticks = np.arange(np.ceil(left), np.floor(right))
    lat_ticks = np.arange(np.ceil(bot), np.floor(top))
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.yaxis.tick_left()
    print("added ticks")
    ax.xaxis.tick_bottom()

    # TODO: start of zebra frame?
    # https://stackoverflow.com/questions/44273365/color-axis-spine-with-multiple-colors-using-matplotlib
    # colors=["b","r","lightgreen","gold"]
    # x=[0,.25,.5,.75,1]
    # y=[0,0,0,0,0]
    # points = np.array([x, y]).T.reshape(-1, 1, 2)
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # lc = LineCollection(segments,colors=colors, linewidth=2,
    #                                transform=ax.get_xaxis_transform(), clip_on=False )
    # ax.add_collection(lc)
    # ax.spines["bottom"].set_visible(False)
    # ax.set_xticks(x)


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


def map_img(img, bbox=None, pad_pct=0.0, ax=None, crs=None, **imshow_kwargs):
    import cartopy.crs as ccrs

    if crs is None:
        crs = ccrs.PlateCarree()

    if ax is None:
        fig = plt.figure(figsize=(8, 3))
        ax = fig.add_subplot(1, 1, 1, projection=crs)

    if bbox is None:
        bbox = _get_rio_bbox(img)

    extent_img = _padded_extent(bbox, 0.0)
    axim = ax.imshow(img, transform=crs, extent=extent_img, origin="upper", **imshow_kwargs)
    extent = _padded_extent(bbox, pad_pct)
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
    proj,
    length,
    location=(0.5, 0.05),
    scalewidth=3,
    units="km",
    m_per_unit=1000,
    zorder=1,
    lw=5,
):
    """
    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    proj is the projection the axes are in
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
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

    # find lat/lon center to find best UTM zone
    x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
    # Projection in metres
    utm = ccrs.UTM(latlon.utm_from_lon((x0 + x1) / 2))
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
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
    )
