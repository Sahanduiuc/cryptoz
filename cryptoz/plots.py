import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cryptoz import utils

plt.style.use("ggplot")


# colormaps
###########

def discrete_cmap(colors):
    return mcolors.ListedColormap(colors)


def continuous_cmap(colors, N=100):
    return mcolors.LinearSegmentedColormap.from_list('cont_cmap', colors, N=N)


def discrete_norm(cmap, bounds):
    return mcolors.BoundaryNorm(bounds, cmap.N)


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, midpoint=None, vmin=None, vmax=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def midpoint_norm(midpoint):
    return MidpointNormalize(midpoint=midpoint)


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


def unravel_index(df):
    min_idx, min_col = np.unravel_index(np.nanargmin(df.values), df.values.shape)
    max_idx, max_col = np.unravel_index(np.nanargmax(df.values), df.values.shape)
    return min_idx, min_col, max_idx, max_col


def heatmap(df, cmap, norm=None, col_ranker=None, idx_ranker=None, figsize=None):
    """Plot a matrix heatmap"""
    print(utils.describe_df(df, flatten=True))

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

    plt.show()


def get_vmin_vmax(df, symmetric=False):
    vmax = df.max().max()
    if symmetric:
        if vmax > 0:
            vmin = -vmax
        else:
            vmin = df.min().min()
    else:
        vmin = df.min().min()
    return vmin, vmax


def evolution(df, cmap=plt.cm.Spectral, norm=None, vmin=None, vmax=None, rank=None, 
                sentiment=lambda sr: sr.mean(), show_markers=None, describe=False, figsize=None):
    """Combine multiple time series into a heatmap and plot"""
    if describe:
        print(utils.describe_df(df, flatten=False))

    if rank is not None:
        if isinstance(rank, str):
            if rank == 'correlation':
                rank = lambda sr: -np.corrcoef(df[df.columns[0]], sr)[0, 1]
            elif rank == 'last':
                rank = lambda sr: -sr.iloc[-1]
            elif rank == 'mean':
                rank = lambda sr: -sr.mean()
        ranks = [rank(df[c]) for c in df.columns]
        columns, _ = zip(*sorted(zip(df.columns, ranks), key=lambda x: x[1]))
        df = df[list(columns)]
    index = trunk_dt_index(df.index)
    columns = df.columns

    plt.close('all')
    if figsize is None:
        figsize = (14, len(columns) * 0.45)
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.pcolor(df.transpose(), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)

    x = np.arange(len(index)) + 0.5
    y = np.arange(len(columns)) + 0.5

    if show_markers == 'global':
        min_x, min_y, max_x, max_y = unravel_index(df)
        min_x, min_y, max_x, max_y = x[min_x], y[min_y], x[max_x], y[max_y]
        ax.plot(min_x, min_y, marker='v', markersize=10, markerfacecolor='white', 
            markeredgewidth=1, markeredgecolor='black')
        ax.plot(max_x, max_y, marker='^', markersize=10, markerfacecolor='white', 
            markeredgewidth=1, markeredgecolor='black')
    elif show_markers == 'local':
        for i, j in list(zip(np.nanargmin(df.values, axis=0), range(len(df.columns)))):
            ax.plot(x[i], y[j], marker='v', markersize=10, markerfacecolor='white', 
                markeredgewidth=1, markeredgecolor='black')
        for i, j in list(zip(np.nanargmax(df.values, axis=0), range(len(df.columns)))):
            ax.plot(x[i], y[j], marker='^', markersize=10, markerfacecolor='white', 
                markeredgewidth=1, markeredgecolor='black')

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
    sentiment_sr = df.apply(sentiment, axis=1)
    cax_top.pcolor([sentiment_sr], cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
    cax_top.set_yticks([0.5], minor=False)
    cax_top.set_yticklabels(["sentiment"], minor=False)
    plt.setp(cax_top.get_xticklabels(), visible=False)

    plt.grid(False)

    plt.show()
