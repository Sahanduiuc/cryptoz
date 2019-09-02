import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
import pandas as pd

from cryptoz import utils

##########################################
# Normalization

def get_vmin_vmax(df):
    vmin = df.min().min()
    vmax = df.max().max()
    if vmin < 0 and vmax > 0:
        # Symmetric around 0 by default
        vmax = max([abs(vmin), abs(vmax)])
        vmin = -vmax
    return vmin, vmax


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, midpoint=None, vmin=None, vmax=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


##########################################
# Colormaps

def discrete_cmap(colors):
    return mcolors.ListedColormap(colors)


def continuous_cmap(colors, N=255):
    return mcolors.LinearSegmentedColormap.from_list('cont_cmap', colors, N=N)


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


def cmap_to_colorscale(cmap, vmin, vmax, norm=None, cmap_discrete=False, cmap_range=None, num=255):
    assert(isinstance(cmap, (str, mcolors.ListedColormap, mcolors.LinearSegmentedColormap)))
    
    if isinstance(cmap, str):
        # Transform string to colormap object
        cmap = mpl.cm.get_cmap(cmap)
    
    if norm is None:
        # You can do any color normalization (midpoint, asymmetric, discrete, etc.)
        # By default, just normalize into 0..1
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # Select a range from the colormap
    # For example, one can select [1, 0] to reverse the colormap
    if cmap_range is None:
        # By default, it selects the whole range
        cmap_range = [0, 1]

    colorscale = []
    converter = mcolors.colorConverter.to_rgb

    # Go through min(z) to max(z), normalize and convert to RGB
    # For each RGB, also prepend its relative position on the color scale (which is 0..1)
    
    # The higher num, the clearer are color bounds
    z_space = np.linspace(vmin, vmax, num=num)
    last_z = None
    for i, z in enumerate(z_space):
        # Normalize z-value, map it into colormap and transform to RGB
        cmap_val = cmap(utils.rescale_single(norm(z), [0, 1], cmap_range))
        rgb = tuple(map(np.uint8, np.array(converter(cmap_val))*255))
        rgb_str = 'rgb'+str((rgb[0], rgb[1], rgb[2]))
        # Scale position must be between 0 and 1
        # 0.1 means let first 10% of z values have this color
        scale_pos = i*1.0/(len(z_space)-1)
        if cmap_discrete:
            # For clear bounds between colors
            if last_z is not None:
                if z != last_z[1]:
                    colorscale.append([last_z[0], rgb_str])
            last_z = (scale_pos, z)
        # For example, [[0, 'green'], [0.5, 'red'], [1.0, 'rgb(0, 0, 255)']]
        colorscale.append([scale_pos, rgb_str])

    return colorscale


##########################################
# Heatmaps

def time_heatmap(df, cmap=None, norm=None, vmin=None, vmax=None, cmap_discrete=False, 
    cmap_range=None, rank_func=None, sentiment_func=lambda sr: sr.mean(), zsmooth=None, 
    describe=False, static=False, column_height=20):
    """Plot a heatmap with time on x-axis and columns on y-axis."""
    
    #########################
    # Configure the dataframe
    
    if describe:
        # Print details of df before plotting
        print(utils.describe(df, flatten=False))
        
    if rank_func is not None:
        # correlation: rank_func = lambda sr: -np.corrcoef(df[df.columns[0]], sr)[0, 1]
        # last: rank_func = lambda sr: -sr.iloc[-1]
        # mean: rank_func = lambda sr: -sr.mean()
        df = df[df.columns[np.argsort([rank_func(df[c]) for c in df.columns])]]
        
    ###########################
    # Configure the color scale
    
    # Choose some good default parameters, since we know the data
    if vmin is None and vmax is None:
        vmin, vmax = get_vmin_vmax(df)
    if vmin is None:
        # The lowest value on the color scale
        vmin = df.min().min()
    if vmax is None:
        # The highest value on the color scale
        vmax = df.max().max()
    if cmap is None and norm is None:
        if vmin < 0 and vmax > 0:
            # Bipolar range
            norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)
        elif vmin <= 0 and vmax <= 0:
            # Negative range only
            cmap_range = [0, 0.5]
        elif vmin >= 0 and vmax >= 0:
            # Positive range only
            cmap_range = [0.5, 1]
    if cmap is None:
        cmap = 'Spectral'

    if isinstance(cmap, (str, mcolors.ListedColormap, mcolors.LinearSegmentedColormap)):
        # If cmap is a matplotlib colormap, transform into plotly's colorscale
        # Note: norm is only valid with matplotlib's colormap
        colorscale = cmap_to_colorscale(cmap, vmin, vmax, norm=norm, 
            cmap_discrete=cmap_discrete, cmap_range=cmap_range)        
    else:
        # other 
        colorscale = cmap
    if zsmooth is None:
        # Smooth the sentiment heatmap only if the colormap is continuous
        if cmap_discrete is False:
            zsmooth = 'fast'
        else:
            zsmooth = False
            

    ########################
    # Configure the subplots
    
    trace1 = go.Heatmap(
        z=df.transpose().values,
        x=df.index,
        y=[c[0]+'/'+c[1] for c in df.columns],
        colorscale=colorscale, 
        zmin=vmin,
        zmax=vmax,
        showscale=True,
        hoverinfo='x+y+z',
        hovertemplate="index: %{x}<br>column: %{y}<br>value: %{z}",
        xaxis="x",
        yaxis="y")
    
    sentiment_df = df.apply(sentiment_func, axis=1)
    trace2 = go.Heatmap(
        z=[sentiment_df.values],
        x=df.index,
        y=['sentiment'],
        colorscale=colorscale, 
        zmin=vmin,
        zmax=vmax,
        zsmooth=zsmooth,
        showscale=False,
        hoverinfo='x+z',
        hovertemplate="index: %{x}<br>value: %{z}",
        xaxis="x",
        yaxis="y2")
    data = [trace1, trace2]

    # Define the hover event 
    def hover_fn(trace, points, state):
        # TODO: One tooltip for both subplots
        pass

    trace1.on_hover(hover_fn)
    
    ######################
    # Configure the layout
    
    # Margins and paddings
    ml = 30
    mr = 30
    mb = 30
    mt = 30
    pad = 10
    
    # Calculate the size of the subplots and the figure as whole
    labels_height = 20
    fixed_height = labels_height + mt + mb
    # Size of the figure
    width = 900
    height = len(df.columns) * column_height + 2 * column_height + fixed_height
    # Range of each subplot (vertically)
    y_domain = [0, 1 - (2 * column_height) / (height - fixed_height)]
    y2_domain = [1 - (column_height) / (height - fixed_height), 1]
        
    # Configure layout
    layout = go.Layout(
        yaxis=dict(
            domain=y_domain
        ),
        yaxis2=dict(
            domain=y2_domain
        ),
        autosize=False,
        width=width,
        height=height,
        margin=go.layout.Margin(
            l=ml,
            r=mr,
            b=mb,
            t=mt,
            pad=pad
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    #################
    # Build the chart

    fig = go.FigureWidget(data=data, layout=layout)
    if static:
        # Show a static PNG image
        fig.show(renderer="png", width=width, height=height)
    else:
        # Return the interactive chart (for Jupyter Lab/Notebook)
        return fig
