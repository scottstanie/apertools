"""Utilities for configuring and generating publication-quality Matplotlib figures."""

from pathlib import Path
import math
import subprocess
import tempfile
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa  (used in user code)

# Global figure parameters
GLOB_FIG_PARAMS = {
    "fontsize": 8,
    "family": "serif",
    "usetex": True,
    "preamble": r"\usepackage{amsmath} \usepackage{times} \usepackage{mathtools}",
    "column_inch": 229.8775 / 72.27,
    "markersize": 24,
    "markercolour": "#AA00AA",
    "fontcolour": "#666666",
    "tickdirection": "out",
    "linewidth": 0.5,
    "ticklength": 2.50,
    "minorticklength": 1.1,
}

# Example color list
COLORS_LIST = ["#2c334f", "#90845b", "#deafda", "#8f605a", "#aaaaaa", "#dddddd"]

# Example colormaps
CMAP_CYCLIC = cc.cm["CET_C1"]
CMAP_CLIPPED = colors.LinearSegmentedColormap.from_list(
    "clipped", CMAP_CYCLIC(np.linspace(0.2, 0.8, 256))
)
CMAP_DIV = CMAP_CLIPPED  # or something like cc.cm['CET_CBD2']
CMAP_MAG = cc.cm["gray"]  # or cc.cm['CET_CBL2']


def initialize_matplotlib() -> None:
    """
    Initialize Matplotlib rcParams for consistent publication-quality plots.

    This function applies a global style defined by the GLOB_FIG_PARAMS dictionary.
    """
    plt.rc("font", size=GLOB_FIG_PARAMS["fontsize"], family=GLOB_FIG_PARAMS["family"])
    plt.rcParams["text.usetex"] = GLOB_FIG_PARAMS["usetex"]
    plt.rcParams["text.latex.preamble"] = GLOB_FIG_PARAMS["preamble"]
    plt.rcParams["legend.fontsize"] = GLOB_FIG_PARAMS["fontsize"]
    plt.rcParams["font.size"] = GLOB_FIG_PARAMS["fontsize"]
    plt.rcParams["axes.linewidth"] = GLOB_FIG_PARAMS["linewidth"]
    plt.rcParams["axes.labelcolor"] = GLOB_FIG_PARAMS["fontcolour"]
    plt.rcParams["axes.edgecolor"] = GLOB_FIG_PARAMS["fontcolour"]
    plt.rcParams["xtick.color"] = GLOB_FIG_PARAMS["fontcolour"]
    plt.rcParams["xtick.direction"] = GLOB_FIG_PARAMS["tickdirection"]
    plt.rcParams["ytick.direction"] = GLOB_FIG_PARAMS["tickdirection"]
    plt.rcParams["ytick.color"] = GLOB_FIG_PARAMS["fontcolour"]
    plt.rcParams["xtick.major.width"] = GLOB_FIG_PARAMS["linewidth"]
    plt.rcParams["ytick.major.width"] = GLOB_FIG_PARAMS["linewidth"]
    plt.rcParams["xtick.minor.width"] = GLOB_FIG_PARAMS["linewidth"]
    plt.rcParams["ytick.minor.width"] = GLOB_FIG_PARAMS["linewidth"]
    plt.rcParams["ytick.major.size"] = GLOB_FIG_PARAMS["ticklength"]
    plt.rcParams["xtick.major.size"] = GLOB_FIG_PARAMS["ticklength"]
    plt.rcParams["ytick.minor.size"] = GLOB_FIG_PARAMS["minorticklength"]
    plt.rcParams["xtick.minor.size"] = GLOB_FIG_PARAMS["minorticklength"]
    plt.rcParams["text.color"] = GLOB_FIG_PARAMS["fontcolour"]


def prepare_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[float, float] = (1.7, 0.8),
    figsizeunit: str = "col",
    sharex: Union[str, bool] = "col",
    sharey: Union[str, bool] = "row",
    bottom: float = 0.10,
    left: float = 0.15,
    right: float = 0.95,
    top: float = 0.95,
    hspace: float = 0.5,
    wspace: float = 0.1,
    remove_spines: bool = True,
    gridspec_kw: Optional[dict] = None,
    subplot_kw: Optional[dict] = None,
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray, None]]:
    """
    Prepare a Matplotlib figure and subplots with publication-ready styling.

    Parameters
    ----------
    nrows : int, optional
        Number of rows of subplots, by default 1.
    ncols : int, optional
        Number of columns of subplots, by default 1.
    figsize : tuple of float, optional
        Base figure size (width, height). The actual figure size will scale
        depending on `figsizeunit`, by default (1.7, 0.8).
    figsizeunit : {'col', 'in'}, optional
        If 'col', multiply the width and height of figsize by `column_inch`
        from GLOB_FIG_PARAMS. If 'in', no scaling is applied, by default 'col'.
    sharex : {'none', 'all', 'row', 'col', bool}, optional
        Controls sharing of properties among x axes, by default 'col'.
    sharey : {'none', 'all', 'row', 'col', bool}, optional
        Controls sharing of properties among y axes, by default 'row'.
    squeeze : bool, optional
        If True, extra dimensions are squeezed out from returned Axes object, by default True.
    bottom : float, optional
        The bottom margin of the subplots, by default 0.10.
    left : float, optional
        The left margin of the subplots, by default 0.15.
    right : float, optional
        The right margin of the subplots, by default 0.95.
    top : float, optional
        The top margin of the subplots, by default 0.95.
    hspace : float, optional
        The amount of height reserved for space between subplots, by default 0.5.
    wspace : float, optional
        The amount of width reserved for space between subplots, by default 0.1.
    remove_spines : bool, optional
        Whether to remove the top and right spines from each subplot, by default True.
    gridspec_kw : dict, optional
        Dictionary with keywords passed to the GridSpec constructor, by default None.
    subplot_kw : dict, optional
        Dictionary with keywords passed to add_subplot, by default None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    axes : matplotlib.axes.Axes or ndarray of Axes or None
        The created Axes object(s). If `nrows` or `ncols` is 0, returns None.
    """
    initialize_matplotlib()

    if figsizeunit == "col":
        scale_factor = GLOB_FIG_PARAMS["column_inch"]
    elif figsizeunit == "in":
        scale_factor = 1.0
    else:
        raise ValueError("figsizeunit must be either 'col' or 'in'.")

    fig_size = (figsize[0] * scale_factor, figsize[1] * scale_factor)
    figprops = {"facecolor": "white", "figsize": fig_size}

    if nrows > 0 and ncols > 0:
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
            gridspec_kw=gridspec_kw,
            subplot_kw=subplot_kw,
            **figprops,
        )
        fig.subplots_adjust(
            bottom=bottom, left=left, right=right, top=top, hspace=hspace, wspace=wspace
        )
    else:
        fig = plt.figure(**figprops)
        axes = None

    if remove_spines and axes is not None:
        for ax in axes.ravel():
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

    return fig, axes


def get_fig_size(
    fig_width_cm: float, fig_height_cm: Optional[float] = None
) -> Tuple[float, float]:
    """
    Convert dimensions in centimeters to inches.

    If no height is given, it is computed using the golden ratio.

    Parameters
    ----------
    fig_width_cm : float
        The width of the figure in centimeters.
    fig_height_cm : float, optional
        The height of the figure in centimeters. If None, uses the golden ratio.

    Returns
    -------
    tuple of float
        The (width, height) in inches.
    """
    if fig_height_cm is None:
        golden_ratio = (1 + math.sqrt(5)) / 2
        fig_height_cm = fig_width_cm / golden_ratio

    width_in = fig_width_cm / 2.54
    height_in = fig_height_cm / 2.54
    return width_in, height_in


def save_fig(
    fig: plt.Figure,
    file_name: str,
    fmt: Optional[str] = None,
    dpi: int = 300,
    tight: bool = True,
) -> None:
    """
    Save a Matplotlib figure in EPS/PNG/PDF format and trim it using system tools.

    The figure is first saved to a temporary file (with optional tight bounding box),
    and then trimmed or cropped (if a suitable external tool is available) and saved
    to the final path.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    file_name : str
        The desired file name (without or with extension).
    fmt : {'eps', 'png', 'pdf'}, optional
        The format in which to save the figure. If None, the format is inferred
        from the `file_name` extension, by default None.
    dpi : int, optional
        Resolution in dots per inch, by default 300.
    tight : bool, optional
        Whether to use bbox_inches='tight' when saving, by default True.

    Raises
    ------
    ValueError
        If the file format is not supported.

    Notes
    -----
    - Requires command-line tools (epstool, convert, pdfcrop) for trimming/cropping.
    - On some systems, you may need to install these tools or modify your PATH.
    """
    if fmt is None:
        # Infer format from extension
        ext = Path(file_name).suffix.lower().strip(".")
        if ext in {"eps", "png", "pdf"}:
            fmt = ext
        else:
            raise ValueError(
                "No valid format was inferred from file_name. "
                "Specify fmt explicitly or provide a file extension of eps/png/pdf."
            )

    if fmt not in {"eps", "png", "pdf"}:
        raise ValueError(f"Unsupported format: {fmt}")

    desired_ext = f".{fmt}"
    if not file_name.endswith(desired_ext):
        file_name += desired_ext

    file_path = Path(file_name).absolute()

    with tempfile.NamedTemporaryFile(suffix=desired_ext, delete=False) as tmp_file:
        tmp_name = tmp_file.name

    # Save figure
    save_kwargs = {"dpi": dpi}
    if tight:
        save_kwargs["bbox_inches"] = "tight"

    fig.savefig(tmp_name, **save_kwargs)

    # Trim/crop
    if fmt == "eps":
        cmd = f'epstool --bbox --copy "{tmp_name}" "{file_path}"'
    elif fmt == "png":
        cmd = f'convert "{tmp_name}" -trim "{file_path}"'
    else:  # fmt == 'pdf'
        cmd = f'pdfcrop "{tmp_name}" "{file_path}"'

    subprocess.run(cmd, shell=True, check=True)

    # Cleanup temporary file
    try:
        Path(tmp_name).unlink()
    except OSError:
        pass
