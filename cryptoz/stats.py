import numpy as np

from cryptoz import utils

import numpy as np
import pandas as pd

from cryptoz import utils


##########################################
# Correlation


def rolling_corr_sr(sr1, sr2, *args, **kwargs):
    return sr1.rolling(*args, **kwargs).corr(other=sr2)


def resampling_corr_sr(sr1, sr2, *args, **kwargs):
    apply_func = lambda sr1, sr2: sr1.corr(other=sr2) if len(sr1.index) > 1 else np.nan
    return utils.resampling_apply(sr1, lambda sr: apply_func(sr, sr2), *args, **kwargs)


def pairwise_apply_corr(df, against_col, apply_func):
    combi_func = lambda cols: [(col, against_col) for col in cols if col != against_col]
    new_df = utils.pairwise_apply(df, combi_func, apply_func)
    # Shorten column names, omit against_col in each column
    new_df.columns = list(map(lambda x: x[0] if x[0] != against_col else x[1], new_df.columns))
    return new_df


def rolling_corr(df, against_col, *args, **kwargs):
    apply_func = lambda sr1, sr2: rolling_corr_sr(sr1, sr2, *args, **kwargs)
    return pairwise_apply_corr(df, against_col, apply_func)


def resampling_corr(df, against_col, *args, **kwargs):
    apply_func = lambda sr1, sr2: resampling_corr_sr(sr1, sr2, *args, *kwargs)
    return pairwise_apply_corr(df, against_col, apply_func)


def expanding_corr(df, against_col, *args, **kwargs):
    apply_func = lambda sr1, sr2: resampling_corr_sr(sr1, sr2, *args, *kwargs)
    return pairwise_apply_corr(df, against_col, apply_func)


######



_rolling_max = lambda ohlc_df: ohlc_df['H'].rolling(window=len(ohlc_df.index), min_periods=1).max()
_dd = lambda ohlc_df: 1 - ohlc_df['L'] / _rolling_max(ohlc_df)


def from_ohlc(ohlc):
    return pd.DataFrame({pair: _dd(ohlc_df) for pair, ohlc_df in ohlc.items()})


# How far are we now from the last max?
_dd_now = lambda ohlc_df: 1 - ohlc_df['C'].iloc[-1] / _rolling_max(ohlc_df).iloc[-1]


def now(ohlc, delta=None):
    return pd.Series({pair: _dd_now(ohlc_df) for pair, ohlc_df in ohlc.items()}).sort_values()


def rolling(ohlc, reducer, *args, **kwargs):
    _dd = from_ohlc(ohlc)
    return utils.rolling_apply(_dd, reducer, *args, **kwargs)


def resampling(ohlc, reducer, *args, **kwargs):
    _dd = from_ohlc(ohlc)
    return utils.resampling_apply(_dd, reducer, *args, **kwargs)


_maxdd_duration = lambda ohlc_df, dd_sr: dd_sr.argmin() - ohlc_df.loc[:dd_sr.argmin(), 'H'].argmax()


def max_duration(ohlc):
    _dd = from_ohlc(ohlc)
    return pd.Series({pair: _maxdd_duration(ohlc[pair], _dd[pair]) for pair in ohlc.keys()}).sort_values()


def _period_details(ohlc_df, group_df):
    """Details of a DD and recovery"""
    if len(group_df.index) > 1:
        window_df = group_df.iloc[1:]  # drawdown starts at first fall
        max = group_df['H'].iloc[0]
        min = window_df['L'].min()
        # DD
        start = window_df.index[0]
        valley = window_df['L'].argmin()
        dd_len = len(window_df.loc[start:valley].index)
        dd = 1 - min / max
        # Recovery
        if len(ohlc_df.loc[group_df.index[-1]:].index) > 1:
            # Recovery finished
            end = window_df.index[-1]
            recovery_len = len(window_df.loc[valley:end].index)
            recovery_rate = dd_len / recovery_len
            recovery = max / min - 1
        else:
            # Not recovered yet
            end = np.nan
            recovery_len = np.nan
            recovery_rate = np.nan
            recovery = np.nan

        return start, valley, end, dd_len, recovery_len, recovery_rate, dd, recovery
    return np.nan


def _details(ohlc_df):
    details_func = lambda group_df: _period_details(ohlc_df, group_df)
    # Everything below last max forms a group
    group_sr = (~((ohlc_df['H'] - _rolling_max(ohlc_df)) < 0)).astype(int).cumsum()
    details = ohlc_df.groupby(group_sr).apply(details_func).dropna().values.tolist()
    columns = ['start', 'valley', 'end', 'dd_len', 'recovery_len', 'recovery_rate', 'dd', 'recovery']
    return pd.DataFrame(details, columns=columns)


def details(ohlc):
    return {pair: _details(ohlc_df) for pair, ohlc_df in ohlc.items()}

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
