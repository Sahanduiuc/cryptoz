import numpy as np

from cryptoz import utils


def corr(df):
    return df.corr()


_apply = lambda sr1, sr2: sr1.corr(other=sr2) if len(sr1.index) > 1 else np.nan


def _rolling_corr(sr1, sr2, window):
    return utils.rolling_apply(sr1, window, lambda sr: _apply(sr, sr2))


def _resampled_corr(sr1, sr2, period):
    return utils.resampled_apply(sr1, period, lambda sr: _apply(sr, sr2))


def rolling_corr(df, window):
    apply_func = lambda sr1, sr2: _rolling_corr(sr1, sr2, window)
    combi_func = utils.combine
    return utils.pairwise_apply(df, combi_func, apply_func)


def resampled_corr(df, period):
    apply_func = lambda sr1, sr2: _resampled_corr(sr1, sr2, period)
    combi_func = utils.combine
    return utils.pairwise_apply(df, combi_func, apply_func)
