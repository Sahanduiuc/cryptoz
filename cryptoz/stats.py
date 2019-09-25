import numpy as np
import pandas as pd

from cryptoz import utils


##########################################
# Correlation

def pairwise_apply_corr(df, against_col, apply_func):
    """Calculate pairwise correlation of each column and the passed column."""
    combi_func = lambda cols: [(col, against_col) for col in cols if col != against_col]
    new_df = utils.pairwise_apply(df, combi_func, apply_func)
    # Shorten column names, omit against_col in each column
    new_df.columns = list(map(lambda x: x[0] if x[0] != against_col else x[1], new_df.columns))
    return new_df


def rolling_corr(df, against_col, backwards=False, *args, **kwargs):
    """Calculate correlation with respect to a column and rolling window."""
    sr2 = df[against_col]
    if backwards:
        apply_func = lambda sr: sr.iloc[::-1].rolling(*args, **kwargs).corr(other=sr2.iloc[::-1]).iloc[::-1]
    else:
        apply_func = lambda sr: sr.rolling(*args, **kwargs).corr(other=sr2)
    return utils.apply(df, apply_func)


def expanding_corr(df, against_col, backwards=False, *args, **kwargs):
    """Calculate correlation with respect to a column and the expanding window."""
    sr2 = df[against_col]
    if backwards:
        apply_func = lambda sr: sr.iloc[::-1].expanding(*args, **kwargs).corr(other=sr2.iloc[::-1]).iloc[::-1]
    else:
        apply_func = lambda sr: sr.expanding(*args, **kwargs).corr(other=sr2)
    return utils.apply(df, apply_func)


def resampling_corr(df, against_col, *args, **kwargs):
    """Calculate correlation with respect to a column and the sample window."""
    apply_func = lambda sr: sr.corr(other=df[against_col]) if len(sr.index) > 1 else np.nan
    return utils.resampling_apply(df, apply_func, *args, **kwargs)


#########################################
# Percentiles

def percentiles_sr(sr, min, max, step):
    index = range(min, max + 1, step)
    return pd.Series({x: np.nanpercentile(sr, x) for x in index})


def percentiles(df, *args):
    return df.apply(lambda sr: percentiles_sr(sr, *args))


#########################################
# Drawdowns and Recovery

def mdd_sr(sr):
    """Calculate max drawdown (MDD) of the time series."""
    # Already expanding, no need for expanding_mdd
    return 1 - sr / sr.expanding().max()


def mdd(df, *args, **kwargs):
    """Max drawdown with respect to all elements."""
    return utils.apply(df, mdd_sr, *args, **kwargs)


def rolling_mdd(df, *args, **kwargs):
    """Max drawdown with respect to a rolling window."""
    f = lambda sr: mdd_sr(sr)[-1]
    return utils.rolling_apply(df, f, *args, **kwargs)


def resampling_mdd(df, *args, **kwargs):
    """Max drawdown with respect to the sample window."""
    return utils.resampling_apply(df, mdd_sr, *args, **kwargs)


def dd_mapreduce_sr(sr, map_func, reduce_func):
    """Split the time series into drawdown ranges and apply mapper and reducer on them.
    
    Mapper should accept `sr` (whole pd.Series), `begin_idx`, `end_idx` parameters.
    Reducer should accept a list of mapper outputs.
    """
    _mdd_sr = mdd_sr(sr)
    diff_sr = (_mdd_sr > 0).astype(int).diff()
    # Indices where drawdown begins
    begin_idxs = diff_sr[diff_sr == 1].index.tolist()
    # Include one more index to the left
    begin_idxs = [diff_sr.index[diff_sr.index.get_loc(idx)-1] for idx in begin_idxs]
    # Indices where recovery ends
    end_idxs = diff_sr[diff_sr == -1].index.tolist()
    if len(begin_idxs) > len(end_idxs):
        # If the last drawdown still lasts, add the recovery date as None
        end_idxs.append(None)
    if len(begin_idxs) > 0 and len(end_idxs) > 0:
        mapped_lst = [map_func(sr, idx1, idx2) for idx1, idx2 in zip(begin_idxs, end_idxs)]
        return reduce_func(mapped_lst)
    else:
        # The rate continuously increases
        return None


def dd_stats(df, map_func, reduce_func):
    """Split each time series into drawdown ranges and aggregate their statistics."""
    return pd.Series([dd_mapreduce_sr(df[c], map_func, reduce_func) for c in df.columns], index=df.columns)


def dd_map_info_sr(sr, begin_idx, end_idx):
    """Return the drawdown and recovery information on a time series range.
    
    A drawdown is a peak-to-trough decline during a specific time period.
    A recovery is an increase from trough to peak equal or higher than the previous peak.
    """
    focus_sr = sr.loc[begin_idx:end_idx]
    valley_idx = focus_sr.idxmin()
    return {
        'start': begin_idx,
        'valley': valley_idx,
        'end': end_idx,
        'dd_duration': valley_idx - begin_idx,
        'rec_duration': end_idx - valley_idx if end_idx is not None else None,
        'dd_rate (%)': (1 - focus_sr.min() / focus_sr.iloc[0]) * 100,
        'rec_rate (%)': (focus_sr.iloc[-1] / focus_sr.min() - 1) * 100 if end_idx is not None else None
    }
    
def dd_info(df, *args, **kwargs):
    """Give the information on drawdowns and recovery for each column."""
    return {c: dd_mapreduce_sr(df[c], dd_map_info_sr, lambda x: pd.DataFrame(x)) for c in df.columns}
    
