import os
from datetime import date
from itertools import repeat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from . import gps
from . import plotting
from apertools.sario import LOS_FILENAME
from apertools.log import get_log

logger = get_log()


def plot_gps_los(
    name,
    insar_mm_list=[],
    labels=None,
    insar_colors=None,
    ref=None,
    start_date=date(2014, 11, 1),
    end_date=None,
    days_smooth=0,
    ax=None,
    ylim=(-3, 3),
    yticks=[-2, 0, 2],
    ylabel="[cm]",
    title="",
    zero_start=False,
    zero_mean=True,
    offset=True,
    lw=5,
    rasterized=True,
    gps_color="#86b251",
    ms=7,
    los_map_file=LOS_FILENAME,
    los_enu_coeffs=None,
    los_da=None,
):
    if labels is None:
        labels = repeat(None, len(insar_mm_list))
    if insar_colors is None:
        insar_colors = repeat(None, len(insar_mm_list))

    df = gps.load_gps_los(
        station_name=name,
        days_smooth=days_smooth,
        reference_station=ref,
        start_date=start_date,
        end_date=end_date,
        zero_start=zero_start,
        zero_mean=zero_mean,
        los_map_file=los_map_file,
        enu_coeffs=los_enu_coeffs,
        los_da=los_da,
    )
    dts = df.index
    day_nums = (dts - dts[0]).days
    # day_nums = _get_day_nums(dts)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(
        df.index,
        df.los,
        marker=".",
        color=gps_color,
        markersize=ms,
        label="GPS",
        rasterized=rasterized,
    )

    for label, insar_mm, c in zip(labels, insar_mm_list, insar_colors):
        insar_cm_day = insar_mm / 365 / 10
        full_defo = insar_cm_day * (dts[-1] - dts[0]).days
        bias = -full_defo / 2 if offset else 0

        ax.plot(dts, bias + day_nums * insar_cm_day, "-", c=c, lw=lw, label=label)

    ax.grid(which="major", alpha=0.5)
    ax.set_xticks(pd.date_range(dts[0], end=dts[-1], freq="365D"))
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)

    ax.set_ylim(ylim)
    return fig, ax


def plot_gps_enu(
    station=None,
    days_smooth=24,
    start_date=None,
    end_date=None,
    colordots=None,
    colorline=None,
    lw=2,
    markersize=5,
    ylim=None,
    nrows=3,
    ncols=1,
    use_proplot=False,
    **subplot_kw,
):
    """Plot the east,north,up components of `station`"""

    def remove_xticks(ax):
        ax.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )

    if nrows == 1:
        ncols = 3
    elif ncols == 3:
        nrows = 1
    assert nrows * ncols == 3, "nrows and ncols must be 1 or 3"

    enu_df = gps.load_station_enu(
        station,
        start_date=start_date,
        end_date=end_date,
        to_cm=True,
    )
    dts = enu_df.index
    (east_cm, north_cm, up_cm) = enu_df[["east", "north", "up"]].T.values

    if use_proplot:
        import proplot as pplt

        print(subplot_kw)

        fig, axes = pplt.subplots(
            nrows=nrows, ncols=ncols, sharex=True, sharey=False, **subplot_kw
        )
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes[0].plot(dts, east_cm, ".", color=colordots, markersize=markersize)
    axes[0].set_ylabel("East [cm]")
    if days_smooth and days_smooth > 0:
        axes[0].plot(
            dts, gps.moving_average(east_cm, days_smooth), color=colorline, lw=lw
        )
    axes[0].grid(True)
    axes[0].set_ylim(ylim)
    # remove_xticks(axes[0])

    axes[1].plot(dts, north_cm, ".", color=colordots, markersize=markersize)
    axes[1].set_ylabel("North [cm]")
    if days_smooth and days_smooth > 0:
        axes[1].plot(
            dts, gps.moving_average(north_cm, days_smooth), color=colorline, lw=lw
        )
    axes[1].grid(True)
    axes[1].set_ylim(ylim)
    # remove_xticks(axes[1])

    axes[2].plot(dts, up_cm, ".", color=colordots, markersize=markersize)
    axes[2].set_ylabel("Up [cm]")
    if days_smooth and days_smooth > 0:
        axes[2].plot(
            dts, gps.moving_average(up_cm, days_smooth), color=colorline, lw=lw
        )
    axes[2].grid(True)
    axes[2].set_ylim(ylim)
    axes.format(xlabel="", yticks=axes[0].get_yticks())
    # remove_xticks(axes[2])

    # fig.suptitle(station)
    # plt.show(block=False)

    return fig, axes


def plot_insar_vs_gps(
    geo_path=None,
    defo_filename="deformation.npy",
    station_name_list=None,
    df=None,
    kind="line",
    reference_station=None,
    **kwargs,
):
    """Make a GPS vs InSAR plot.

    kinds:
        line: plot out full data for each station
        errorbar: predict cumulative value for each station with error bars
        slope: plot gps value vs predicted insar (with 1-1 slope being perfect),
            gives insar error bars

    If reference_station is provided, all columns are centered to that series
        with gps subtracting the gps, insar subtracting the insar
    """

    def _filter_df_by_stations(df, station_name_list, igrams_dir, defo_filename):
        station_name_list = _load_station_list(
            igrams_dir=igrams_dir,
            defo_filename=defo_filename,
            station_name_list=station_name_list,
        )
        select_cols = [
            col for col in df.columns if any(name in col for name in station_name_list)
        ]
        return df[select_cols]

    igrams_dir = os.path.join(geo_path, "igrams")
    if df is None:
        df = create_insar_gps_df(
            geo_path,
            defo_filename=defo_filename,
            # station_name_list=station_name_list,  # For now, filter given names after grabbing all
            reference_station=reference_station,
            **kwargs,
        )
        # window_size=1, days_smooth_insar=5, days_smooth_gps=30):
    if station_name_list is not None:
        df = _filter_df_by_stations(df, station_name_list, igrams_dir, defo_filename)

    return plot_insar_gps_df(df, kind=kind, **kwargs)


def plot_insar_gps_df(
    df, kind="errorbar", grid=True, block=False, velocity=True, **kwargs
):
    """Plot insar vs gps values from dataframe

    kinds:
        line: plot out full data for each station
        errorbar: predict cumulative value for each station with error bars
        slope: plot gps value vs predicted insar (with 1-1 slope being perfect),
            gives insar error bars
    """
    valid_kinds = ("line", "errorbar", "slope")

    # for idx, column in enumerate(columns):
    if kind == "errorbar":
        fig, axes = _plot_errorbar_df(df, velocity=velocity, **kwargs)
    elif kind == "line":
        fig, axes = _plot_line_df(df, **kwargs)
    elif kind == "slope":
        fig, axes = _plot_slope_df(df, **kwargs)
    else:
        raise ValueError("kind must be in: %s" % valid_kinds)
    fig.tight_layout()
    if grid:
        for ax in axes.ravel():
            ax.grid(True)

    plt.show(block=block)
    return fig, axes


def _plot_errorbar_df(df, ylim=None, velocity=True, **kwargs):
    gps_cols, insar_cols, final_gps_vals, final_insar_vals = get_final_gps_insar_values(
        df, velocity=velocity, **kwargs
    )
    gps_stds = [flat_std(df[col].dropna()) for col in df.columns if col in gps_cols]

    fig, axes = plt.subplots(squeeze=False)
    ax = axes[0, 0]
    idxs = range(len(final_gps_vals))
    ax.errorbar(
        idxs, final_gps_vals, gps_stds, marker="o", lw=2, linestyle="", capsize=6
    )
    ax.plot(idxs, final_insar_vals, "rx")

    if velocity:
        ax.set_ylabel("mm/year of LOS displacement")
    else:
        ax.set_ylabel("CM of cumulative LOS displacement")

    labels = [c.replace("_gps", "") for c in gps_cols]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation="vertical", fontsize=12)
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, axes


def _plot_slope_df(df, **kwargs):
    gps_cols, insar_cols, final_gps_vals, final_insar_vals = get_final_gps_insar_values(
        df
    )
    insar_stds = [flat_std(df[col].dropna()) for col in df.columns if col in insar_cols]

    fig, axes = plt.subplots(squeeze=False)
    ax = axes[0, 0]
    ax.errorbar(final_gps_vals, final_insar_vals, yerr=insar_stds, fmt="rx", capsize=6)

    max_val = max(np.max(final_insar_vals), np.max(final_gps_vals))
    min_val = min(np.min(final_insar_vals), np.min(final_gps_vals))
    ax.plot(np.linspace(min_val, max_val), np.linspace(min_val, max_val), "g")
    ax.set_ylabel("InSAR predicted cumulative CM")
    ax.set_xlabel("GPS cumulative CM")
    return fig, axes


def _plot_smoothed(ax, df, column, days_smooth, marker, lw=3, color=None, shift=0):
    series = df[column].dropna().rolling(days_smooth, min_periods=1, center=True).mean()
    return series, ax.plot(
        series.index,
        series + shift,
        marker,
        linewidth=lw,
        label="%s day smoothed %s" % (days_smooth, column),
        color=color,
    )


def _plot_line_df(
    df,
    ylim=None,
    share=True,
    days_smooth_gps=None,
    days_smooth_insar=None,
    figsize=(10, 5),
    show_grid=True,
    **kwargs,
):
    """share is used to indicate that GPS and insar will be on same axes"""
    columns = df.columns
    nrows = 1 if share else 2
    fig, axes = plt.subplots(nrows, len(columns) // 2, figsize=figsize, squeeze=False)

    gps_idxs = np.where(["gps" in col for col in columns])[0]
    insar_idxs = np.where(["insar" in col for col in columns])[0]

    for idx, column in enumerate(columns):
        if "insar" in column:
            marker = "rx"
            alpha = 1.0
            ax_idx = np.where(insar_idxs == idx)[0][0] if share else idx
        else:
            marker = "b."
            alpha = 0.5
            ax_idx = np.where(gps_idxs == idx)[0][0] if share else idx

        ax = axes.ravel()[ax_idx]
        ax.plot(
            df.index,
            df[column].fillna(method="ffill"),
            marker,
            label=column,
            alpha=alpha,
        )
        # ax.plot(df.index, df[column], marker, label=column, alpha=0.5)

        ax.set_title(column)
        if ylim is not None:
            ax.set_ylim(ylim)
        xticks = ax.get_xticks()
        ax.set_xticks([xticks[0], xticks[len(xticks) // 2], xticks[-1]])
        if days_smooth_gps and "gps" in column:
            _plot_smoothed(ax, df, column, days_smooth_gps, "b-")
        if days_smooth_insar and "insar" in column:
            _plot_smoothed(ax, df, column, days_smooth_insar, "r-")

        ax.set_ylabel("Cum. Defo. [cm]")
        ax.legend()
        ax.grid(show_grid)

    # axes.ravel()[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    return fig, axes


def plot_all_stations(
    df,
    df_diff=None,
    ncols=2,
    days_smooth_gps=30,
    ylim=None,
    share=False,
    figsize=(10, 10),
    lw_insar=3,
    lw_gps=3,
    alpha_gps=1.0,
    color_insar=None,
    color_gps=None,
    rasterize_gps=True,
    outname=None,
    plot_std=False,
    std_nsigma=2,
    alpha_std=0.5,
    color_std=None,
    abc=False,
    return_lines=False,
    shift_gps=False,
    return_rms=False,
    return_shifts=False,
):
    import proplot as pplt

    names = sorted(
        set(
            n.replace("_gps", "").replace("_insar", "").replace("_diff", "")
            for n in df.columns
        )
    )
    # nplots = len(df_diff.index)
    nplots = len(names)
    # fig, axes = plt.subplots(
    fig, axes = pplt.subplots(
        nrows=int(np.ceil(nplots / ncols)),
        ncols=ncols,
        figsize=figsize,
        sharex=share,
        sharey=share,
        abc=abc,
        # figsize=(4 * ncols, nplots)
    )
    # axes = axes.ravel()
    rms_dict = {}
    shift_dict = {}
    # for idx, name in enumerate(df_diff.index):
    for idx, name in enumerate(names):
        lines = []
        # ax = axes.ravel()[idx]
        ax = axes[idx]
        gps_col, insar_col = f"{name}_gps", f"{name}_insar"
        if shift_gps:
            shift = _find_rms_shift(df[gps_col], df[insar_col], gps_rolling_window=360)
        else:
            shift = 0
        shift_dict[name] = shift

        l1 = ax.plot(
            df.index,
            df[gps_col] + shift,
            marker=".",
            linestyle="none",
            ms=1,
            alpha=alpha_gps,
            color=color_gps,
            rasterized=rasterize_gps,
        )
        lines.append(l1)
        if days_smooth_gps and days_smooth_gps > 1:
            gps_sm, l2 = _plot_smoothed(
                ax,
                df,
                gps_col,
                days_smooth_gps,
                "-",
                lw=lw_gps,
                color=color_gps,
                shift=shift,
            )
            lines.append(l2)

        df_nona = df[[insar_col]].dropna()
        # print(f"{plot_std = }, {c in df.columns}")
        if plot_std and f"{name}_std" in df.columns:
            std = df.loc[df_nona.index, f"{name}_std"]
            lf = ax.fill_between(
                df_nona.index,
                df_nona[insar_col] - std_nsigma * std,
                df_nona[insar_col] + std_nsigma * std,
                alpha=alpha_std,
                color=color_std,
            )
            lines.append(lf)

        l3 = ax.plot(
            df_nona.index,
            df_nona[insar_col],
            # marker="o",
            # ms=1,
            lw=lw_insar,
            color=color_insar,
        )
        lines.append(l3)
        # print(df[insar_col].dropna())
        # ax.legend()
        # ax.grid()
        ax.set_title(name)

        gps_sm = df[gps_col].rolling(30, min_periods=1).mean()
        d = (gps_sm + shift - df[insar_col]).dropna()
        rms_dict[name] = (rms(d), maxabs(d))

    axes.format(grid=True, ylim=ylim, ylabel="cm", xlabel="")
    if outname:
        fig.savefig(outname)

    ret = [fig, axes]
    if return_lines:
        ret.append(lines)
    if return_rms:
        ret.append(rms_dict)
    if return_shifts:
        ret.append(shift_dict)
    return ret


def rms(x):
    return np.sqrt(np.nanmean(x**2))


def maxabs(x):
    return np.nanmax(np.abs(x))


def _find_rms_shift(gps, insar, gps_rolling_window=30, search_range=(-1, 1)):
    """Find RMS-minimizing shift between two series. Add this to GPS"""
    gps_sm = gps.rolling(gps_rolling_window, min_periods=1).mean()
    diff = (insar - gps_sm).dropna()
    return np.clip(diff.mean(), *search_range)


def plot_stations_on_image(
    df_diff,
    ax,
    ms=10,
    marker="X",
    add_labels=True,
):
    plotted_points = []
    for idx, n in enumerate(df_diff.index):
        lon, lat = gps.station_lonlat(n)
        ax.plot(lon, lat, ms=ms, marker=marker)
        plotted_points.append((lon, lat))
    if add_labels:
        for idx, (lon, lat) in enumerate(plotted_points):
            ax.annotate(df_diff.index[idx], (lon, lat), fontsize=10)
    return plotted_points


def plot_gps_east_by_loc(
    defo_filename,
    igrams_dir,
    start_date=None,
    end_date=None,
    cmap_name="seismic_r",
    **plot_kwargs,
):
    enu_df = create_gps_enu_df(
        defo_filename=defo_filename,
        igrams_dir=igrams_dir,
        end_date=end_date,
        start_date=start_date,
    )
    east_cols = [col for col in enu_df if "east" in col]
    east_df = enu_df[east_cols]

    df_locations = create_station_location_df(east_df)
    df_final_vals = get_final_east_values(east_df)
    df_merged = df_locations.join(df_final_vals)

    labels = [
        "{}: {:.2f}".format(stat, val)
        for stat, val in zip(df_merged.index, df_merged["east"])
    ]
    xs = df_merged["lon"]
    ys = df_merged["lat"]
    vals = df_merged["east"]
    # Note: reversed for ascending path so that red = west movement = toward satellite
    cmap = plotting.make_shifted_cmap(
        cmap_name=cmap_name, vmax=vals.max(), vmin=vals.min()
    )

    first_date, last_date = east_df.index[0], east_df.index[-1]
    title = "east GPS movement from {} to {}".format(first_date, last_date)
    fig, axes = _plot_latlon_with_labels(
        xs, ys, vals, labels, title=title, cmap=cmap, **plot_kwargs
    )
    return df_merged, fig, axes


def plot_residuals_by_loc(
    df, which="diff", title=None, fig=None, ax=None, plot_scatter=True, **plot_kwargs
):
    """Takes a timeseries df and plots the final values at their lat/lons

    df should be the timeseries df with "date" as index created by create_insar_gps_df
    `which` argument is "diff","gps","insar", where 'diff' takes (gps - insar)
    """

    def _build_labels(df_merged, values):
        """Inclue station name and final value in label"""
        return [
            "{}:\n{:.2f}".format(name, val)
            for name, val in zip(df_merged.index, values)
        ]

    df_merged, values = _get_residuals(df, which)

    print("Total abs values summed:", total_abs_error(values))
    print("RMS value:", rms(values))

    xs = df_merged["lon"]
    ys = df_merged["lat"]
    labels = _build_labels(df_merged, values)
    if title is None:
        title = "Final values of %s by location" % which
    fig, ax = _plot_latlon_with_labels(
        xs,
        ys,
        values,
        labels,
        title="",
        fig=fig,
        ax=ax,
        plot_scatter=plot_scatter,
        **plot_kwargs,
    )
    return df_merged, fig, ax


def _plot_latlon_with_labels(
    xs,
    ys,
    values,
    labels,
    title="",
    fig=None,
    ax=None,
    plot_scatter=True,
    **plot_kwargs,
):
    fig, ax = plotting.get_fig_ax(fig, ax)

    if plot_scatter:
        axim = ax.scatter(xs, ys, c=values, zorder=10, **plot_kwargs)
        fig.colorbar(axim)

    for label, x, y in zip(labels, xs, ys):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(-10, 10),
            textcoords="offset points",
            # ha='right',
            # va='bottom',
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        )
    ax.set_title(title)
    return fig, ax
