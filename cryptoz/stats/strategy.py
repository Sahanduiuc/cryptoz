import numpy as np
import pandas as pd


def _BB(sr, window, std_n):
    # Price tends to return back to mean
    rollmean_sr = sr.rolling(window=window, min_periods=1).mean()
    rollstd_sr = sr.rolling(window=window, min_periods=1).std()
    upper_band_sr = rollmean_sr + std_n * rollstd_sr
    lower_band_sr = rollmean_sr - std_n * rollstd_sr
    return upper_band_sr, lower_band_sr


def BB_crossover(ohlc_df, window, std_n):
    upper_band_sr, lower_band_sr = _BB(ohlc_df['C'], window, std_n)
    vector = np.where(ohlc_df['C'] > upper_band_sr, 1, 0)
    vector = np.where(ohlc_df['C'] < lower_band_sr, -1, vector)
    return pd.Series(vector, index=ohlc_df.index)


def apply(ohlc, strategy_func):
    # strategy_func must create a vector of 1, 0, -1 (long, wait, short)
    return pd.DataFrame({pair: strategy_func(ohlc_df) for pair, ohlc_df in ohlc.items()})
