import numpy as np
import pandas as pd

from cryptoz import utils

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
