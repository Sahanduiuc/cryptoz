import numpy as np

from cryptoz import utils


def corr(df):
    return df.corr()


_apply = lambda sr1, sr2: sr1.corr(other=sr2) if len(sr1.index) > 1 else np.nan


def _rolling_corr(sr1, sr2, *args, **kwargs):
    return utils.rolling_apply(sr1, lambda sr: _apply(sr, sr2), *args, **kwargs)


def _resample_corr(sr1, sr2, *args, **kwargs):
    return utils.resample_apply(sr1, lambda sr: _apply(sr, sr2), *args, **kwargs)


def rolling_corr(df, *args, **kwargs):
    apply_func = lambda sr1, sr2: _rolling_corr(sr1, sr2, *args, **kwargs)
    combi_func = utils.combine
    return utils.pairwise_apply(df, combi_func, apply_func)


def resample_corr(df, *args, **kwargs):
    apply_func = lambda sr1, sr2: _resample_corr(sr1, sr2, *args, *kwargs)
    combi_func = utils.combine
    return utils.pairwise_apply(df, combi_func, apply_func)
