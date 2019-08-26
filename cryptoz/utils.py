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
    """Return generate descriptive statistics as a new dataframe."""
    if flatten:
        df = pd.DataFrame(df.values.flatten())
    return df.describe().transpose()


##########################################
# Windows


def rolling_apply(df, func, backwards=False, *args, **kwargs):
    """Apply a function on a rolling window. PAST ONLY."""
    # Everything after this point of time
    if backwards: return df.iloc[::-1].rolling(*args, **kwargs).apply(func, raw=False).iloc[::-1]
    # Everything before this point of time
    else: return df.rolling(*args, **kwargs).apply(func, raw=False)


def expanding_apply(df, func, backwards=False, *args, **kwargs):
    """Apply a function on the expanding window. EITHER PAST OR FUTURE ONLY."""
    if backwards: return df.iloc[::-1].expanding(*args, **kwargs).apply(func, raw=False).iloc[::-1]
    else: return df.expanding(*args, **kwargs).apply(func, raw=False)


def resampling_apply(df, func, *args, **kwargs):
    """Resample the time series and apply a function on each sample. PAST AND FUTURE.
    
    For example, downsample `df` into 1h bins and apply `func` on each bin.
    """
    period_index = pd.Series(df.index, index=df.index).resample(*args, **kwargs).first().dropna()
    grouper = pd.Series(1, index=period_index.values).reindex(df.index).fillna(0).cumsum()
    return df.groupby(grouper).apply(lambda df: df.apply(func, raw=False))


##########################################
# Combinations

def product(cols):
    """Cartesian product."""
    return list(itertools.product(cols, repeat=2))


def combine(cols):
    """2-length tuples, in sorted order, no repeated elements."""
    return list(itertools.combinations(cols, 2))


def combine_rep(cols):
    """2-length tuples, in sorted order, with repeated elements."""
    return list(itertools.combinations_with_replacement(cols, 2))


def pairwise_apply(df, combi_func, apply_func):
    """Apply a function on each of the combinations of any two columns."""
    colpairs = combi_func(df.columns)
    return pd.DataFrame({(col1, col2): apply_func(df[col1], df[col2]) for col1, col2 in colpairs})


##########################################
# Normalization

def normalize_sr(sr, method):
    """Normalize the array."""
    if method == 'max':
        return sr / sr.max()
    if method == 'minmax':
        return (sr - sr.min()) / (sr.max() - sr.min())
    if method == 'mean':
        return (sr - sr.mean()) / (sr.max() - sr.min())
    if method == 'std':
        return (sr - sr.mean()) / sr.std()


def normalize(df, method, **kwargs):
    """Normalize the dataframe with respect to all elements."""
    f = lambda sr: normalize_sr(sr, method)
    return apply(df, f, **kwargs)


def rolling_normalize(df, method, backwards=False, *args, **kwargs):
    """Normalize the dataframe with respect to a rolling window."""
    f = lambda sr: normalize_sr(sr, method)[-1]
    return rolling_apply(df, f, backwards=backwards, *args, **kwargs)


def expanding_normalize(df, method, backwards=False, *args, **kwargs):
    """Normalize the dataframe with respect to the expanding window."""
    f = lambda sr: normalize_sr(sr, method)[-1]
    return expanding_apply(df, f, backwards=backwards, *args, **kwargs)


def resampling_normalize(df, method, *args, **kwargs):
    """Normalize the dataframe with respect to the sample window."""
    f = lambda sr: normalize_sr(sr, method)
    return resampling_apply(df, f, *args, **kwargs)


##########################################
# Rescaling

def rescale_sr(sr, to_range, from_range=None):
    """Rescale the array."""
    if from_range is None:
        min1, max1 = sr.min(), sr.max()
    else:
        min1, max1 = from_range
    min2, max2 = to_range
    range1 = max1 - min1
    range2 = max2 - min2
    return (sr - min1) * range2 / range1 + min2


def rescale(df, to_range, from_range=None, **kwargs):
    """Rescale the dataframe with respect to all elements."""
    f = lambda sr: rescale_sr(sr, to_range, from_range=from_range)
    return apply(df, f, **kwargs)


def rolling_rescale(df, to_range, from_range=None, backwards=False, *args, **kwargs):
    """Rescale the dataframe with respect to a rolling window."""
    f = lambda sr: rescale_sr(sr, to_range, from_range=from_range)[-1]
    return rolling_apply(df, f, backwards=backwards, *args, **kwargs)


def expanding_rescale(df, to_range, from_range=None, backwards=False, *args, **kwargs):
    """Rescale the dataframe with respect to the expanding window."""
    f = lambda sr: rescale_sr(sr, to_range, from_range=from_range)[-1]
    return expanding_apply(df, f, backwards=backwards, *args, **kwargs)


def resampling_rescale(df, method, *args, **kwargs):
    """Rescale the dataframe with respect to the sample window."""
    f = lambda sr: rescale_sr(sr, method)
    return resampling_apply(df, f, *args, **kwargs)


def rescale_dynamic_range(df, from_range, to_range):
    """Rescale the dataframe dynamically.
    
    A similar idea as for `rescale` but from_range can take a tuple of dataframes.
    """
    min1, max1 = from_range
    min2, max2 = to_range
    return (df - min1) * (max2 - min2) / (max1 - min1) + min2


def reverse_scale(df, **kwargs):
    """Reverse the scale of the dataframe."""
    f = lambda a: np.nanmin(a) + np.nanmax(a) - a
    return apply(df, f, **kwargs)
