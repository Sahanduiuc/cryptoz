import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("ggplot")


def discrete_cmap(bounds, colors):
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def continuous_cmap(colors, midpoint=None, N=6):
    # Create continuous cmap with/without midpoint
    cmap = mcolors.LinearSegmentedColormap.from_list('cont_cmap', colors, N=N)
    if midpoint is not None:
        norm = MidpointNormalize(midpoint=midpoint)
    else:
        norm = mcolors.Normalize()
    return cmap, norm


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


def time_series(df, cmap=None, ranker=None, ncols=3, figsize=None):
    """Display each time series as a line chart"""
    print(pd.DataFrame(df.values.flatten()).describe().transpose())

    nsubplots = len(df.columns)
    nrows = math.ceil(nsubplots / ncols)
    if ranker is not None:
        _, columns = zip(*sorted(enumerate(df.columns), key=lambda x: ranker(df[x[1]])))
    else:
        columns = df.columns

    plt.close('all')
    if figsize is None:
        figsize = (4 * ncols, 2.5 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if (i < nsubplots):
            sr = df[columns[i]]
            x = list(range(len(sr.index)))
            y = sr.bfill().values

            if cmap is None:
                color = 'grey'
            else:
                color = cmap(i / nsubplots)
            alpha = 0.5
            gradient_fill(ax, x, y, color, alpha, ylim=False)

            nticks = 4
            tick_interval = math.ceil(len(df.index) / nticks)
            ax.set_xticks(x[::tick_interval])
            ax.set_xticklabels(trunk_dt_index(df.index)[::tick_interval])
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.set_xlim((min(x), max(x)))
            ax.set_ylim((min(y), max(y)))

            ax.set_title(columns[i])
            ax.grid(False)
        else:
            fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


def matrix(df, norm=plt.Normalize(), cmap=plt.cm.GnBu_r, figsize=None):
    """Display matrix as a heatmap"""
    print(pd.DataFrame(df.values.flatten()).describe().transpose())

    plt.close('all')
    if figsize is None:
        figsize = (len(df.columns) * 0.5 + 0.5, len(df.index) * 0.48)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolor(df, cmap=cmap, norm=norm, vmin=df.min().min(), vmax=df.max().max())
    ax.set_yticks(np.arange(len(df.index)) + 0.5, minor=False)
    ax.set_xticks(np.arange(len(df.columns)) + 0.5, minor=False)
    ax.set_yticklabels(df.index, minor=False)
    ax.set_xticklabels(df.columns, minor=False)

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


def evolution(df, norm=plt.Normalize(), cmap=plt.cm.GnBu_r, figsize=None):
    """Combine multiple time series into a heatmap"""
    print(pd.DataFrame(df.values.flatten()).describe().transpose())
    x = trunk_dt_index(df.index)
    y = df.columns
    df = df.transpose()

    plt.close('all')
    if figsize is None:
        figsize = (14, len(df.index) * 0.4)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolor(df, cmap=cmap, norm=norm, vmin=df.min().min(), vmax=df.max().max())
    # yticks
    yticks = np.arange(len(y)) + 0.5
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(y, minor=False)
    ax.invert_yaxis()
    # xticks
    nticks = 6
    tick_interval = math.ceil(len(x) / nticks)
    xticks = (np.arange(len(x)) + 0.5)
    xticks = xticks[::tick_interval]
    ax.set_xticks(xticks, minor=False)
    ax.set_xticklabels(x[::tick_interval], minor=False)

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


def depth(orderbooks, colors=None, ranker=None, ncols=3, figsize=None):
    """Display order books as depth graphs"""
    print(pd.DataFrame(np.array(list(orderbooks.values())).flatten()).describe().transpose())

    pairs = list(orderbooks.keys())
    nsubplots = len(orderbooks.keys())
    nrows = math.ceil(nsubplots / ncols)
    if ranker is not None:
        _, pairs = zip(*sorted(enumerate(pairs), key=lambda x: ranker(orderbooks[x[1]])))

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
