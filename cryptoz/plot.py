import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("ggplot")


# colormaps
###########

def discrete_cmap(colors):
    return mcolors.ListedColormap(colors)


def continuous_cmap(colors, N=100):
    return mcolors.LinearSegmentedColormap.from_list('cont_cmap', colors, N=N)


def discrete_norm(cmap, bounds):
    return mcolors.BoundaryNorm(bounds, cmap.N)


def midpoint_norm(midpoint):
    return MidpointNormalize(midpoint=midpoint)


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def combine_cmaps(cm1, cm2, scale1, scale2):
    colors = ()
    for cm, scale in [(cm1, scale1), (cm2, scale2)]:
        _colors = cm(np.linspace(0., 1, int(128 / abs(scale[1] - scale[0]))))
        scale = np.array(scale)
        scale *= len(_colors)
        scale = scale.round().astype(int)
        scale = scale.tolist()
        _colors = _colors[slice(*scale)]
        colors += (_colors,)
    colors = np.vstack(colors)
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    return mymap


# Plotting
##########


# https://stackoverflow.com/questions/29321835/is-it-possible-to-get-color-gradients-under-curve-in-matplotlb
def gradient_fill(ax, x, y, color, alpha, ylim=False):
    line, = ax.plot(x, y, color=color)
    zorder = line.get_zorder()

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]
    if ylim:
        ymin, ymax = ylim
    else:
        ymin, ymax = min(y), max(y)
    xmin, xmax = min(x), max(x)
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax], origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    return line, im


def trunk_dt_index(index):
    """Trunk datetime index"""
    tdiff = int((index[-1] - index[0]).total_seconds())

    if tdiff <= 24 * 60 * 60:  # 12:45
        return [i.strftime("%H:%M") for i in index]
    elif tdiff <= 7 * 24 * 60 * 60:  # Tue 12:45
        return [i.strftime("%a %H:%M") for i in index]
    elif tdiff <= 30 * 24 * 60 * 60:  # Wed 11
        return [i.strftime("%a %d") for i in index]
    elif tdiff <= 365 * 24 * 60 * 60:  # Jun 11
        return [i.strftime("%b %d") for i in index]
    else:  # Jun 2018
        return [i.strftime("%b %Y") for i in index]


def hist_matrix(df, bins=20, cmap=None, norm=None, ranker=None, axvlines=None, ncols=3, figsize=None):
    """Plot group of histograms"""
    print(pd.DataFrame(df.values.flatten()).describe().transpose())

    nsubplots = len(df.columns)
    nrows = math.ceil(nsubplots / ncols)
    if ranker is not None:
        ranks = [ranker(df[c]) for c in df.columns]
        columns, _ = zip(*sorted(zip(df.columns, ranks), key=lambda x: x[1]))
    else:
        columns = df.columns

    plt.close('all')
    if figsize is None:
        figsize = (4 * ncols, 2.5 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if (i < nsubplots):
            sr = df[columns[i]].copy()

            hist, _bins = np.histogram(sr, bins=bins)
            width = np.diff(_bins)
            center = (_bins[:-1] + _bins[1:]) / 2
            if cmap is None:
                colors = 'grey'
            else:
                if norm is None:
                    norm = plt.Normalize()
                colors = cmap(norm(_bins))

            ax.bar(center, hist, color=colors, align='center', width=width, alpha=0.5)

            if axvlines is None:
                ax.axvline(sr.mean(), color='black', lw=1)
            else:
                for axvline in axvlines:
                    if callable(axvline):
                        ax.axvline(axvline(sr), color='black', lw=1)
                    else:
                        ax.axvline(axvline, color='black', lw=1)

            ax.set_title(columns[i])
            ax.grid(False)
        else:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


def timesr_matrix(df, bands=None, ranker=None, ncols=3, figsize=None):
    """Plot group of line charts"""
    print(pd.DataFrame(df.values.flatten()).describe().transpose())

    nsubplots = len(df.columns)
    nrows = math.ceil(nsubplots / ncols)
    if ranker is not None:
        ranks = [ranker(df[c]) for c in df.columns]
        columns, _ = zip(*sorted(zip(df.columns, ranks), key=lambda x: x[1]))
    else:
        columns = df.columns

    plt.close('all')
    if figsize is None:
        figsize = (4 * ncols, 2.5 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if (i < nsubplots):
            sr = df[columns[i]].copy()
            sr.index = trunk_dt_index(sr.index)
            x = list(range(len(sr.index)))
            y = sr.bfill().values

            min_x, max_x, min_y, max_y = np.min(x), np.max(x), np.min(y), np.max(y)
            offset = 0.2 * (max_y - min_y)
            min_y -= offset
            max_y += offset

            gradient_fill(ax, x, y, 'grey', 0.5, ylim=(min_y, max_y))

            if bands is not None:
                upper_band = bands[0][columns[i]]
                lower_band = bands[1][columns[i]]
                ax.fill_between(x, upper_band, lower_band, color='steelblue', alpha=0.2)

            ax.plot(sr.values.argmin(), sr.min(), marker='x', markersize=10, color='black')
            ax.plot(sr.values.argmax(), sr.max(), marker='x', markersize=10, color='black')

            ax.set_xticks([sr.values.argmin(), sr.values.argmax()])
            ax.set_xticklabels([sr.argmin(), sr.argmax()])
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            ax.set_xlim((min_x, max_x))
            ax.set_ylim((min_y, max_y))

            ax.set_title(columns[i])
            ax.grid(False)
        else:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


def unravel_index(df):
    min_idx, min_col = np.unravel_index(np.nanargmin(df.values), df.values.shape)
    max_idx, max_col = np.unravel_index(np.nanargmax(df.values), df.values.shape)
    return min_idx, min_col, max_idx, max_col


def heatmap(df, col_ranker=None, idx_ranker=None, norm=None, cmap=plt.cm.GnBu, figsize=None):
    """Plot a matrix heatmap"""
    print(pd.DataFrame(df.values.flatten()).describe().transpose())

    if col_ranker is not None:
        ranks = [col_ranker(df.loc[:, c]) for c in df.columns]
        columns, _ = zip(*sorted(zip(df.columns, ranks), key=lambda x: x[1]))
        df = df[list(columns)]
    if idx_ranker is not None:
        ranks = [idx_ranker(df.loc[i, :]) for i in df.index]
        index, _ = zip(*sorted(zip(df.index, ranks), key=lambda x: x[1]))
        df = df.reindex(list(index))

    plt.close('all')
    if figsize is None:
        figsize = (len(df.columns) * 0.5 + 0.5, len(df.index) * 0.48)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolor(df, cmap=cmap, norm=norm, vmin=df.min().min(), vmax=df.max().max())

    x = np.arange(len(df.columns)) + 0.5
    y = np.arange(len(df.index)) + 0.5

    min_idx, min_col, max_idx, max_col = unravel_index(df)
    min_x, min_y, max_x, max_y = x[min_col], y[min_idx], x[max_col], y[max_idx]
    ax.plot(min_x, min_y, marker='x', markersize=10, color='black')
    ax.plot(max_x, max_y, marker='x', markersize=10, color='black')

    ax.set_xticks(x, minor=False)
    ax.set_yticks(y, minor=False)
    ax.set_xticklabels(df.columns, minor=False)
    ax.set_yticklabels(df.index, minor=False)

    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.xticks(rotation=45)

    plt.grid(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.35, pad=0.15)

    plt.colorbar(im, cax=cax)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    plt.show()


def evolution(df, ranker='correlation', cmap=plt.cm.GnBu_r, norm=None, reducer=lambda sr: sr.mean(), figsize=None):
    """Combine multiple time series into a heatmap and plot"""
    print(pd.DataFrame(df.values.flatten()).describe().transpose())

    if ranker is not None:
        if isinstance(ranker, str):
            if ranker == 'correlation':
                ranker = lambda sr: -np.corrcoef(df[df.columns[0]], sr)[0, 1]
        ranks = [ranker(df[c]) for c in df.columns]
        columns, _ = zip(*sorted(zip(df.columns, ranks), key=lambda x: x[1]))
        df = df[list(columns)]
    index = trunk_dt_index(df.index)
    columns = df.columns

    plt.close('all')
    if figsize is None:
        figsize = (14, len(columns) * 0.45)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolor(df.transpose(), cmap=cmap, norm=norm, vmin=df.min().min(), vmax=df.max().max())

    x = np.arange(len(index)) + 0.5
    y = np.arange(len(columns)) + 0.5

    min_x, min_y, max_x, max_y = unravel_index(df)
    min_x, min_y, max_x, max_y = x[min_x], y[min_y], x[max_x], y[max_y]
    ax.plot(min_x, min_y, marker='x', markersize=10, color='black')
    ax.plot(max_x, max_y, marker='x', markersize=10, color='black')

    # xticks
    nticks = 6
    tick_interval = math.ceil(len(x) / nticks)
    ax.set_xticks(x[::tick_interval], minor=False)
    ax.set_xticklabels(index[::tick_interval], minor=False)

    # yticks
    ax.set_yticks(y, minor=False)
    ax.set_yticklabels(columns, minor=False)
    ax.invert_yaxis()

    divider = make_axes_locatable(ax)
    cax_right = divider.append_axes("right", size=0.35, pad=0.15)
    plt.colorbar(im, cax=cax_right)

    cax_top = divider.append_axes("top", size=0.35, pad=0.15)
    sentiment_sr = df.apply(reducer, axis=1)
    cax_top.pcolor([sentiment_sr], cmap=cmap, norm=norm, vmin=df.min().min(), vmax=df.max().max())
    cax_top.set_yticks([0.5], minor=False)
    cax_top.set_yticklabels(["sentiment"], minor=False)
    plt.setp(cax_top.get_xticklabels(), visible=False)

    plt.grid(False)

    # Turn off all the ticks
    ax = plt.gca()

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    plt.show()


def depth(orderbooks, colors=None, ranker=None, ncols=3, figsize=None):
    """Plot depth graphs from order books"""
    print(pd.DataFrame(np.array(list(orderbooks.values())).flatten()).describe().transpose())

    pairs = list(orderbooks.keys())
    nsubplots = len(orderbooks.keys())
    nrows = math.ceil(nsubplots / ncols)
    if ranker is not None:
        ranks = [ranker(orderbooks[p]) for p in pairs]
        pairs, _ = zip(*sorted(zip(pairs, ranks), key=lambda x: x[1]))

    plt.close('all')
    if figsize is None:
        figsize = (4 * ncols, 2.5 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    if colors is None:
        colors = ('grey', 'grey')

    for i, ax in enumerate(axes.flat):
        if (i < nsubplots):
            orderbook = orderbooks[pairs[i]]
            bids = orderbook[orderbook > 0]
            asks = np.abs(orderbook[orderbook < 0])
            split = len(bids.index)

            x = orderbook.index
            y = np.abs(orderbook.values)

            min_x = min(x)
            max_x = max(x)
            min_y = 0
            max_y = max(y)

            c1, c2 = colors
            gradient_fill(ax, x[:split], bids.values, c1, 0.5, ylim=(min_y, max_y))
            gradient_fill(ax, x[split:], asks.values, c2, 0.5, ylim=(min_y, max_y))

            ax.set_xlim((min_x, max_x))
            ax.set_ylim((min_y, max_y))

            ax.set_xticks([x[0], x[split], x[-1]])
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            ax.set_title(pairs[i])
            ax.grid(False)
        else:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


def quantiles(df, cmap=plt.cm.GnBu_r, norm=None, agg_func=lambda sr: sr.mean(), figsize=None):
    print(pd.DataFrame(df.values.flatten()).describe().transpose())

    perc_index = range(0, 101, 5)
    nummap_perc_sr = pd.Series({x: np.nanpercentile(nummap_sr, x) for x in perc_index})
    benchmark_perc_sr = pd.Series({x: np.nanpercentile(benchmark_sr, x) for x in perc_index})

    fig, ax = plt.subplots()
    ax.plot(benchmark_perc_sr, color='lightgrey')
    ax.plot(nummap_perc_sr, color='darkgrey')
    ax.fill_between(perc_index,
                    nummap_perc_sr,
                    benchmark_perc_sr,
                    where=nummap_perc_sr > benchmark_perc_sr,
                    facecolor='lime',
                    interpolate=True)
    ax.fill_between(perc_index,
                    nummap_perc_sr,
                    benchmark_perc_sr,
                    where=nummap_perc_sr < benchmark_perc_sr,
                    facecolor='orangered',
                    interpolate=True)
    diff_sr = nummap_perc_sr - benchmark_perc_sr
    ax.plot(diff_sr.idxmax(), nummap_perc_sr.loc[diff_sr.idxmax()], marker='x', markersize=10, color='black')
    ax.plot(diff_sr.idxmin(), nummap_perc_sr.loc[diff_sr.idxmin()], marker='x', markersize=10, color='black')
    plt.show()
