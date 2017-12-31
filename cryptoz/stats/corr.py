import numpy as np

from cryptoz import utils


def apply(df):
    return df.corr()


def _rolling(sr1, sr2, *args, **kwargs):
    return sr1.rolling(*args, **kwargs).corr(other=sr2)


def _resampling(sr1, sr2, *args, **kwargs):
    _apply = lambda sr1, sr2: sr1.corr(other=sr2) if len(sr1.index) > 1 else np.nan
    return utils.resampling_apply(sr1, lambda sr: _apply(sr, sr2), *args, **kwargs)


def rolling(df, *args, **kwargs):
    apply_func = lambda sr1, sr2: _rolling(sr1, sr2, *args, **kwargs)
    combi_func = utils.combine
    return utils.pairwise_apply(df, combi_func, apply_func)


def resampling(df, *args, **kwargs):
    apply_func = lambda sr1, sr2: _resampling(sr1, sr2, *args, *kwargs)
    combi_func = utils.combine
    return utils.pairwise_apply(df, combi_func, apply_func)
