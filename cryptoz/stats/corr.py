import numpy as np

from cryptoz import utils


def apply(df):
    return df.corr()


def _rolling(sr1, sr2, *args, **kwargs):
    return sr1.rolling(*args, **kwargs).corr(other=sr2)


def _resampling(sr1, sr2, *args, **kwargs):
    _apply = lambda sr1, sr2: sr1.corr(other=sr2) if len(sr1.index) > 1 else np.nan
    return utils.resampling_apply(sr1, lambda sr: _apply(sr, sr2), *args, **kwargs)


def pairwise_apply(df, against_col, apply_func):
    combi_func = lambda cols: [(col, against_col) for col in cols if col != against_col]
    new_df = utils.pairwise_apply(df, combi_func, apply_func)
    # Shorten column names, omit against_col in each column
    new_df.columns = list(map(lambda x: x[0] if x[0] != against_col else x[1], new_df.columns))
    return new_df


def rolling(df, against_col, *args, **kwargs):
    apply_func = lambda sr1, sr2: _rolling(sr1, sr2, *args, **kwargs)
    return pairwise_apply(df, against_col, apply_func)



def resampling(df, against_col, *args, **kwargs):
    apply_func = lambda sr1, sr2: _resampling(sr1, sr2, *args, *kwargs)
    return pairwise_apply(df, against_col, apply_func)
