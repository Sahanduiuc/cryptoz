import itertools

import numpy as np
import pandas as pd


# DF

def describe_df(df, flatten=False):
    if flatten:
        df = pd.DataFrame(df.values.flatten())
    return df.describe().transpose()


def to_df(group, column):
    return pd.DataFrame({pair: df[column] for pair, df in group.items()})


def to_sr(group, column, reducer):
    return to_df(group, column).apply(reducer)


def cut_df(df, delta):
    return df[df.index > df.index.max() - delta]


def cut(group, *args):
    return {k: cut_df(ohlc_df, *args) for k, ohlc_df in group.items()}


# Combinations

def product(cols):
    """AA AB AC BA BB BC CA CB CC"""
    return list(itertools.product(cols, repeat=2))


def combine(cols):
    """AB AC AD BC BD CD"""
    return list(itertools.combinations(cols, 2))


def combine_rep(cols):
    """AA AB AC BB BC CC"""
    return list(itertools.combinations_with_replacement(cols, 2))


# Apply on DF

def apply(df, func, axis=0):
    """Apply either on columns, rows or both"""
    if axis is None:
        flatten = df.values.flatten()
        reshaped = func(flatten).reshape(df.values.shape)
        return pd.DataFrame(reshaped, columns=df.columns, index=df.index)
    else:
        return df.apply(lambda sr: func(sr.values), axis=axis)


def resampling_apply(df, func, *args, **kwargs):
    """Apply on resampled data"""
    period_index = pd.Series(df.index, index=df.index).resampling(*args, **kwargs).first()
    grouper = pd.Series(1, index=period_index.values).reindex(df.index).fillna(0).cumsum()
    res_sr = df.groupby(grouper).apply(func)
    res_sr.index = period_index.index
    return res_sr


def rolling_apply(df, func, *args, **kwargs):
    """Apply on rolling window"""
    return df.rolling(*args, **kwargs).apply(func)


def expanding_apply(df, func, backwards=False, **kwargs):
    """Apply on expanding window"""
    if backwards:
        return df.iloc[::-1].expanding(**kwargs).apply(func).iloc[::-1]
    else:
        return df.expanding(**kwargs).apply(func)


def pairwise_apply(df, combi_func, apply_func):
    """Apply on column combinations"""
    colpairs = combi_func(df.columns)
    return pd.DataFrame({col1 + '-' + col2: apply_func(df[col1], df[col2]) for col1, col2 in colpairs})


# Normalization

def _normalize(a, method):
    """Feature scaling"""
    if method == 'max':
        return a / np.nanmax(a)
    if method == 'minmax':
        return (a - np.nanmin(a)) / (np.nanmax(a) - np.nanmin(a))
    if method == 'mean':
        return (a - np.nanmean(a)) / (np.nanmax(a) - np.nanmin(a))
    if method == 'std':
        return (a - np.nanmean(a)) / np.nanstd(a)


def normalize(df, method, axis=0):
    """Takes into account past and future. df is normalized on columns, rows or both."""
    f = lambda a: _normalize(a, method)
    return apply(df, f, axis=axis)


def rolling_normalize(df, method, *args, **kwargs):
    """Normalization isolated to fixed rolling window"""
    f = lambda a: _normalize(a, method)[-1]
    return rolling_apply(df, f, *args, **kwargs)


def expanding_normalize(df, method, *args, **kwargs):
    """Normalization isolated to past only"""
    f = lambda a: _normalize(a, method)[-1]
    return expanding_apply(df, f, *args, **kwargs)


# Rescaling

def _rescale(a, to_scale, from_scale=None):
    """[-10, 0, 10] becomes [0, 0.5, 1] (or any other scale)"""
    a = a.copy()
    if from_scale is not None:
        min1, max1 = from_scale
    else:
        min1, max1 = np.nanmin(a), np.nanmax(a)
    min2, max2 = to_scale
    range1 = max1 - min1
    range2 = max2 - min2
    return (a - min1) * range2 / range1 + min2


def rescale(df, to_scale, from_scale=None, axis=0):
    f = lambda a: _rescale(a, to_scale, from_scale=from_scale)
    return apply(df, f, axis=axis)


def rolling_rescale(df, to_scale, from_scale=None, **kwargs):
    f = lambda a: _rescale(a, to_scale, from_scale=from_scale)[-1]
    return rolling_apply(df, f, **kwargs)


def expanding_rescale(df, to_scale, from_scale=None, **kwargs):
    f = lambda a: _rescale(a, to_scale, from_scale=from_scale)[-1]
    return expanding_apply(df, f, **kwargs)


def range_rescale(df, range1, range2):
    """Rescale range1 to range2 for each number in df"""
    min1, max1 = range1
    min2, max2 = range2
    return (df - min1) * (max2 - min2) / (max1 - min1) + min2


def reverse_scale(df, axis=0):
    """[1, 30, 100] becomes [100, 70, 1]"""
    f = lambda a: np.nanmin(a) + np.nanmax(a) - a
    return apply(df, f, axis=axis)


def combine_scales(df1, df2, operation, to_scale, from_scale=None, axis=0):
    """Combine two scales into one"""
    rescaled1 = rescale(df1, to_scale, from_scale=from_scale, axis=axis)
    rescaled2 = rescale(df2, to_scale, from_scale=from_scale, axis=axis)
    if operation == 'add':
        # moderate enhancement
        operation = lambda df1, df2: df1 + df2
    elif operation == 'multiply':
        # strong enhancement
        operation = lambda df1, df2: df1 * df2
    elif operation == 'subtract':
        # moderate suppression
        operation = lambda df1, df2: (df1 - df2).abs()
    elif operation == 'divide':
        # strong suppression
        operation = lambda df1, df2: (df1 / df2 - df2 / df1).abs()
    return rescale(operation(rescaled1, rescaled2), to_scale, axis=axis)


def trunk(df, limits):
    """Trunk everything above limits"""
    df = df.copy()
    _min, _max = limits
    df[df < _min] = _min
    df[df > _max] = _max
    return df


# Classification

def _classify(a, cuts, bipolar=False):
    """
    Divide array into ranges and assign each one an incrementing number
    cuts = [2, 4]: [1, 2, 3, 4, 5] -> [0, 1, 1, 2, 2]
    """
    if bipolar:
        _a = np.abs(a)
    b = _a.copy()
    cuts = [np.min(_a)] + cuts + [np.max(_a)]
    ranges = list(zip(cuts, cuts[1:]))
    for i, r in enumerate(ranges):
        _min, _max = r
        if i < len(ranges) - 1:
            b[(_a >= _min) & (_a < _max)] = i
        else:
            b[(_a >= _min) & (_a <= _max)] = i
    if bipolar:
        b[a < 0] *= -1
    return b


def classify(df, cuts, bipolar=False, axis=0):
    f = lambda a: _classify(a, cuts, bipolar=bipolar)
    return apply(df, f, axis=axis)
