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


##########################################
# Drawdown

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


def single_dd_info_sr(sr, begin_idx, end_idx):
    """Return the drawdown and recovery information on a time series range.
    
    A drawdown is a peak-to-trough decline during a specific time period.
    A recovery is an increase from trough to peak equal or higher than the previous peak.
    """
    focus_sr = sr[begin_idx:end_idx]
    valley_idx = focus_sr.idxmin()
    return {
        'start': begin_idx,
        'valley': valley_idx,
        'end': end_idx,
        'dd_duration': valley_idx - begin_idx,
        'rec_duration': end_idx - valley_idx if end_idx is not None else pd.Timedelta(0),
        'dd_rate (%)': (1 - focus_sr.min() / focus_sr.iloc[0]) * 100,
        'rec_rate (%)': (focus_sr.iloc[-1] / focus_sr.min() - 1) * 100 if end_idx is not None else None
    }

def dd_info_sr(sr):
    """Split the time series into drawdown ranges and extract information from them."""
    _mdd_sr = mdd_sr(sr)
    diff_sr = (_mdd_sr > 0).astype(int).diff()
    # Indices where drawdown begins
    begin_idxs = diff_sr[diff_sr == 1].index.tolist()
    # Include one more index to the left
    begin_idxs = [diff_sr.index[diff_sr.index.get_loc(idx)-1] for idx in begin_idxs]
    # Indices where recovery ends
    end_idxs = diff_sr[diff_sr == -1].index.tolist()
    if len(begin_idxs) > 0 and len(end_idxs) > 0:
        if len(begin_idxs) > len(end_idxs):
            # If the last drawdown still lasts, add the recovery date as None
            end_idxs.append(None)
        return pd.DataFrame([single_dd_info_sr(sr, idx1, idx2) for idx1, idx2 in zip(begin_idxs, end_idxs)])
    else:
        return None
    
def dd_info(df, *args, **kwargs):
    """Give the information on drawdowns and recovery for each column."""
    return {c: dd_info_sr(df[c]) for c in df.columns}

#######



def _percentiles(sr, min, max, step):
    index = range(min, max + 1, step)
    return pd.Series({x: np.nanpercentile(sr, x) for x in index})


def percentiles(df, *args):
    return df.apply(lambda sr: _percentiles(sr, *args))



#######


import numpy as np
import pandas as pd


def safe_divide(a, b):
    if b == 0:
        return np.nan
    return a / b


# Returns to equity
_e = lambda r: (r.replace(to_replace=np.nan, value=0) + 1).cumprod()

# Total earned/lost
_total = lambda e: e.iloc[-1] / e.iloc[0] - 1

trades = lambda r: (r != 0).sum().item()  # np.int64 to int
profits = lambda r: (r > 0).sum()
losses = lambda r: (r < 0).sum()
winrate = lambda r: safe_divide(profits(r), trades(r))
lossrate = lambda r: safe_divide(losses(r), trades(r))

profit = lambda r: _total(_e(r))
avggain = lambda r: r[r > 0].mean()
avgloss = lambda r: -r[r < 0].mean()
expectancy = lambda r: safe_divide(profit(r), trades(r))
maxdd = lambda r: 1 - (_e(r) / _e(r).expanding(min_periods=1).max()).min()


def sharpe(r, nperiods=None):
    res = safe_divide(r.mean(), r.std())
    if nperiods is not None:
        res *= (nperiods ** 0.5)
    return res


def sortino(r, nperiods=None):
    res = safe_divide(r.mean(), r[r < 0].std())
    if nperiods is not None:
        res *= (nperiods ** 0.5)
    return res


def _summary(r):
    summary_sr = r.describe()
    summary_sr.index = pd.MultiIndex.from_tuples([('distribution', i) for i in summary_sr.index])

    summary_sr.loc[('performance', 'profit')] = profit(r)
    summary_sr.loc[('performance', 'avggain')] = avggain(r)
    summary_sr.loc[('performance', 'avgloss')] = avgloss(r)
    summary_sr.loc[('performance', 'winrate')] = winrate(r)
    summary_sr.loc[('performance', 'expectancy')] = expectancy(r)
    summary_sr.loc[('performance', 'maxdd')] = maxdd(r)

    summary_sr.loc[('risk/return profile', 'sharpe')] = sharpe(r)
    summary_sr.loc[('risk/return profile', 'sortino')] = sortino(r)

    return summary_sr


def summary(ohlc):
    return pd.DataFrame({pair: _summary(ohlc_df['C'].pct_change().fillna(0)) for pair, ohlc_df in ohlc.items()})


def score_matrix(ohlc):
    from cryptoz import score

    summary_df = summary(ohlc)
    summary_df.index = summary_df.index.droplevel()
    return score.apply(summary_df, axis=1) # index-local score
