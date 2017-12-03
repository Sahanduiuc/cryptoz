import numpy as np

def _apply(a):
    """Rescale series into [-1, 1]"""
    min_score, max_score = -1, 1
    scores = a.copy()
    old_range = np.nanmax(scores) - np.nanmin(scores)
    new_range = max_score - min_score
    if old_range == 0:
        scores *= 0
        scores += min_score
    else:
        scores = (scores - np.nanmin(scores)) * new_range / old_range + min_score
    return scores


def apply(df):
    """Past and future based"""
    return df.apply(lambda sr: _apply(sr.values))


def rolling_apply(df, *args, **kwargs):
    """Past based"""
    from cryptoz import utils

    return utils.rolling_apply(df, lambda a: _apply(a)[-1], *args, **kwargs)


def reverse(score_df):
    return score_df.min() + score_df.max() - score_df


def add(scoreA_df, scoreB_df):
    """Scores enhance each other"""
    return apply(scoreA_df + scoreB_df)


def diff(scoreA_df, scoreB_df):
    """Scores diminish each other"""
    return apply((scoreA_df - scoreB_df).abs())
