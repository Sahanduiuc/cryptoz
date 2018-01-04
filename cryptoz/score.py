import numpy as np
import pandas as pd


def _apply(a):
    """Rescale series into [0, 1]"""
    min_score, max_score = 0, 1
    scores = a.copy()
    old_range = np.nanmax(scores) - np.nanmin(scores)
    new_range = max_score - min_score
    if old_range == 0:
        scores *= 0
        scores += min_score
    else:
        scores = (scores - np.nanmin(scores)) * new_range / old_range + min_score
    return scores


def apply(df, axis=None):
    """Past, present and future based"""
    if axis is None:
        # axis None: global score
        flatten = df.values.flatten()
        reshaped = _apply(flatten).reshape(df.values.shape)
        return pd.DataFrame(reshaped, columns=df.columns, index=df.index)
    else:
        # axis 0: column-local score
        # axis 1: index-local score
        return df.apply(lambda sr: _apply(sr.values), axis=axis)


def rolling_apply(df, *args, **kwargs):
    """Past and present based"""
    from cryptoz import utils

    # Rolling through index -> axis 0 forced
    return utils.rolling_apply(df, lambda a: _apply(a)[-1], *args, **kwargs)


def reverse(score_df):
    return score_df.min() + score_df.max() - score_df


def add(scoreA_df, scoreB_df, axis=None):
    """Scores enhance each other"""
    return apply(scoreA_df + scoreB_df, axis=axis)


def diff(scoreA_df, scoreB_df, axis=None):
    """Scores diminish each other"""
    return apply((scoreA_df - scoreB_df).abs(), axis=axis)
