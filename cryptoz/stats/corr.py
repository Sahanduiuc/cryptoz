import numpy as np


def corr(df):
    return df.corr()


def _rolling_corr(sr1, sr2, window):
    from cryptoz import utils

    apply_func = lambda sr: sr.corr(other=sr2) if len(sr.index) > 1 else np.nan
    return utils.rolling_apply(sr1, window, apply_func)


def _resampled_corr(sr1, sr2, period):
    from cryptoz import utils

    apply_func = lambda sr: sr.corr(other=sr2) if len(sr.index) > 1 else np.nan
    return utils.resampled_apply(sr1, period, apply_func)


def rolling_corr(df, window):
    from cryptoz import utils

    apply_func = lambda sr1, sr2: _rolling_corr(sr1, sr2, window)
    combi_func = utils.combine
    return utils.pairwise_apply(df, combi_func, apply_func)


def resampled_corr(df, period):
    from cryptoz import utils

    apply_func = lambda sr1, sr2: _resampled_corr(sr1, sr2, period)
    combi_func = utils.combine
    return utils.pairwise_apply(df, combi_func, apply_func)
