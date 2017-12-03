import numpy as np

def _apply(a, min_score, max_score):
    """Rescale series into [min_score, max_score]"""
    scores = a.copy()
    old_range = np.nanmax(scores) - np.nanmin(scores)
    new_range = max_score - min_score
    if old_range == 0:
        scores *= 0
        scores += min_score
    else:
        scores = (scores - np.nanmin(scores)) * new_range / old_range + min_score
    return scores


def apply(df, *args, **kwargs):
    """Past and future based"""
    return df.apply(lambda sr: _apply(sr.values, *args, **kwargs))


def safe_apply(df, *args, **kwargs):
    """Past based"""
    apply_func = lambda a: _apply(a, *args, **kwargs)[-1]
    return df.rolling(window=len(df.index), min_periods=1).apply(apply_func)


def reverse(score_df):
    return score_df.min() + score_df.max() - score_df


def add(scoreA_df, scoreB_df):
    """Scores enhance each other"""
    return apply(scoreA_df + scoreB_df, -1, 1)


def diff(scoreA_df, scoreB_df):
    """Scores diminish each other"""
    return apply((scoreA_df - scoreB_df).abs(), -1, 1)
