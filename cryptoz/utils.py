import itertools

import numpy as np
import pandas as pd

##########################################
# Build DataFrame

def select_window(df, window):
    """Select the latest `window` records from the dataframe.

    `window` should of type `pd.Timedelta`.
    """
    return df[df.index > df.index.max() - window].copy()


def build_from_dict(df_dict, column, window=None):
    """Build a dataframe from the dictionary of dataframes.
    
    Selects a column (and optionally a window) from each one and then stacks them.
    """
    df = pd.DataFrame({pair: df[column] for pair, df in df_dict.items()})
    if window is not None:
        df = select_window(df, window)
    return df


def describe(df, flatten=False):
    """Return generate descriptive statistics of *df* as a new dataframe."""
    if flatten:
        df = pd.DataFrame(df.values.flatten())
    return df.describe().transpose()


##########################################
# Windows

def apply(df, func, axis=0):
    """Apply a function either on columns, rows or both. PAST AND FUTURE.
    
    Each function must operate on an NumPy array, not pd.Series.
    """
    if axis is None:
        # Apply on both axes
        flatten = df.values.flatten()
        reshaped = func(flatten).reshape(df.values.shape)
        return pd.DataFrame(reshaped, columns=df.columns, index=df.index)
    else:
        # Apply on either horizontal or vertical axis
        return df.apply(lambda sr: func(sr.values), axis=axis, raw=False)


def resampling_apply(df, func, *args, **kwargs):
    """Resample the time series and apply a function on each sample. PAST AND FUTURE.
    
    For example, downsample `df` into 1h bins and apply `func` on each bin.
    """
    period_index = pd.Series(df.index, index=df.index).resampling(*args, **kwargs).first()
    grouper = pd.Series(1, index=period_index.values).reindex(df.index).fillna(0).cumsum()
    res_sr = df.groupby(grouper).apply(func, raw=False)
    res_sr.index = period_index.index
    return res_sr


def rolling_apply(df, func, *args, **kwargs):
    """Apply a function on a rolling window. PAST ONLY."""
    return df.rolling(*args, **kwargs).apply(func, raw=False)


def expanding_apply(df, func, backwards=False, **kwargs):
    """Apply a function on the expanding window. EITHER PAST OR FUTURE ONLY."""
    if backwards:
        # Everything before this point of time
        return df.iloc[::-1].expanding(**kwargs).apply(func, raw=False).iloc[::-1]
    else:
        # Everything after this point of time
        return df.expanding(**kwargs).apply(func, raw=False)


##########################################
# Combinations

def product(cols):
    """Cartesian product.
    
    ABC -> AA AB AC BA BB BC CA CB CC
    """
    return list(itertools.product(cols, repeat=2))


def combine(cols):
    """2-length tuples, in sorted order, no repeated elements.
    
    ABC -> AB AC AD BC BD CD
    """
    return list(itertools.combinations(cols, 2))


def combine_rep(cols):
    """2-length tuples, in sorted order, with repeated elements.
    
    ABC -> AA AB AC BB BC CC
    """
    return list(itertools.combinations_with_replacement(cols, 2))


def pairwise_apply(df, combi_func, apply_func):
    """Apply a function on each of the combinations of any two columns."""
    colpairs = combi_func(df.columns)
    return pd.DataFrame({(col1, col2): apply_func(df[col1], df[col2]) for col1, col2 in colpairs})


##########################################
# Normalization

def normalize_arr(a, method):
    """Normalize the array.
    
    max:    [1, 2, 3, 4, 5] -> [0.2, 0.4, 0.6, 0.8, 1]
    minmax: [1, 2, 3, 4, 5] -> [0, 0.25, 0.5, 0.75, 1]
    mean:   [1, 2, 3, 4, 5] -> [-0.5, -0.25, 0, 0.25, 0.5]
    std:    [1, 2, 3, 4, 5] -> [-1.41421356, -0.70710678, 0, 0.70710678, 1.41421356]
    """
    if method == 'max':
        return a / np.nanmax(a)
    if method == 'minmax':
        return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))
    if method == 'mean':
        return (a - np.nanmean(a)) / (np.nanmax(a) - np.nanmin(a))
    if method == 'std':
        return (a - np.nanmean(a)) / np.nanstd(a)


def normalize(df, method, **kwargs):
    """Normalize the dataframe with respect to all elements."""
    f = lambda a: normalize_arr(a, method)
    return apply(df, f, **kwargs)


def rolling_normalize(df, method, *args, **kwargs):
    """Normalize the dataframe with respect to a rolling window."""
    f = lambda a: normalize_arr(a, method)[-1]
    return rolling_apply(df, f, *args, **kwargs)


def expanding_normalize(df, method, *args, **kwargs):
    """Normalize the dataframe with respect to the expanding window."""
    f = lambda a: normalize_arr(a, method)[-1]
    return expanding_apply(df, f, *args, **kwargs)


##########################################
# Rescaling

def rescale_arr(a, to_range, from_range=None):
    """Rescale the array.
    
    [1, 2, 3, 4, 5], [0, 1]          -> [0, 0.25, 0.5, 0.75, 1]
    [1, 2, 3, 4, 5], [0, 1], [0, 10] -> [0.1, 0.2, 0.3, 0.4, 0.5]
    """
    if from_range is None:
        min1, max1 = np.nanmin(a), np.nanmax(a)
    else:
        min1, max1 = from_range
    min2, max2 = to_range
    range1 = max1 - min1
    range2 = max2 - min2
    return (a - min1) * range2 / range1 + min2


def rescale(df, to_range, from_range=None, **kwargs):
    """Rescale the dataframe with respect to all elements."""
    f = lambda a: rescale_arr(a, to_range, from_range=from_range)
    return apply(df, f, **kwargs)


def rolling_rescale(df, to_range, from_range=None, **kwargs):
    """Rescale the dataframe with respect to a rolling window."""
    f = lambda a: rescale_arr(a, to_range, from_range=from_range)[-1]
    return rolling_apply(df, f, **kwargs)


def expanding_rescale(df, to_range, from_range=None, **kwargs):
    """Rescale the dataframe with respect to the expanding window."""
    f = lambda a: rescale_arr(a, to_range, from_range=from_range)[-1]
    return expanding_apply(df, f, **kwargs)


def rescale_dynamic_range(df, from_range, to_range):
    """Rescale the dataframe dynamically.
    
    A similar idea as for `rescale` but from_range can take a tuple of dataframes.
    """
    min1, max1 = from_range
    min2, max2 = to_range
    return (df - min1) * (max2 - min2) / (max1 - min1) + min2


def reverse_scale(df, **kwargs):
    """Reverse the scale of the dataframe.
    
    [1, 2, 3, 4, 7] -> [7, 6, 5, 4, 1]
    """
    f = lambda a: np.nanmin(a) + np.nanmax(a) - a
    return apply(df, f, **kwargs)


def trunk(df, limits):
    """Trunk every element in the dataframe that exceeds some limits"""
    df = df.copy()
    _min, _max = limits
    df[df < _min] = _min
    df[df > _max] = _max
    return df


##########################################
# Classification

def apply_thresholds_arr(a, levels=None, increment=False):
    """For each number in the array, assign it the nearest level such that abs(number) >= abs(level).

    Without levels, it just rounds each number.
    original:                  [-6.5, -5, -2, -0.1, 7, 8, 0, 1.1, 1.0, 4, 5, 222]
    sorted:                    [-6.5, -5, -2, -0.1, 0, 1.0, 1.1, 4, 5, 7, 8, 222]
    levels (-2, 1, 5):                     |            |           |
    sorted output:             [  -2, -2, -2,    0, 0,  1,    1, 1, 5, 5, 5,   5]
    incremented sorted output: [   0,  0,  0,    1, 1,  2,    2, 2, 3, 3, 3,   3]
    """

    if levels is None:
        return list(map(lambda x: round(x), a))
    else:
        # Reverse positive numbers since we want to check bigger numbers first
        levels = list(set(levels + [0]))
        levels = list(sorted(filter(lambda x: x <= 0, levels))) + \
            list(sorted(filter(lambda x: x > 0, levels), reverse=True))
        b = []
        for x in a:
            if np.isnan(x):
                b.append(x)
                continue
            found = False
            for l in levels:
                if (x < 0 and l < 0 and x <= l) or (x > 0 and l > 0 and x >= l):
                    if increment:
                        b.append(sorted(levels).index(l))
                    else:
                        b.append(l)
                    found = True
                    break
            if not found:
                if increment:
                    b.append(sorted(levels).index(0))
                else:
                    b.append(0)
        return b


def apply_thresholds(df, levels=None, increment=False, **kwargs):
    """Apply thresholds to the whole dataframe."""
    f = lambda a: apply_thresholds_arr(a, levels=levels, increment=increment)
    # No need for rolling or expanding windows
    return apply(df, f, **kwargs)
