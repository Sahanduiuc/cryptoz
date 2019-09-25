import unittest

from cryptoz import stats

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.DataFrame(
    {
        'a': [0, 1, 2, 3, 4, 5], 
        'b': [5, 4, 3, 2, 1, 0], 
        'c': [0, 10, 100, 1000, 10000, 100000]
    }, index=[
    datetime(2018, 1, 1, 1, 1, 1),
    datetime(2018, 1, 1, 1, 1, 2),
    datetime(2018, 1, 1, 1, 1, 3),
    datetime(2018, 1, 1, 1, 2, 1),
    datetime(2018, 1, 1, 1, 2, 2),
    datetime(2018, 1, 1, 1, 2, 3)
])

dd_df = pd.DataFrame(
    {
        'a': [1, 2, 3, 2, 1], 
        'b': [3, 2, 1, 2, 3]
    }, index=[
    datetime(2018, 1, 1, 1, 1, 1),
    datetime(2018, 1, 1, 1, 1, 2),
    datetime(2018, 1, 1, 1, 1, 3),
    datetime(2018, 1, 1, 1, 1, 4),
    datetime(2018, 1, 1, 1, 1, 5)
])


class TestStats(unittest.TestCase):

    # Correlation

    def test_rolling_corr(self):
        assert_a = stats.rolling_corr(df, 'c', window=3).values
        assert_b = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [0.9078412990032035, -0.9078412990032034, 1.0],
            [0.9041944301794651, -0.9041944301794651, 1.0],
            [0.9041944301794651, -0.9041944301794651, 1.0],
            [0.9041944301794651, -0.9041944301794651, 1.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_rolling_corr_backwards(self):
        assert_a = stats.rolling_corr(df, 'c', backwards=True, window=3).values
        assert_b = np.array([
            [0.9078412990032065, -0.9078412990032063, 1.0],
            [0.9041944301794651, -0.9041944301794651, 1.0],
            [0.9041944301794651, -0.9041944301794651, 1.0],
            [0.9041944301794651, -0.9041944301794651, 1.0],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_expanding_corr(self):
        assert_a = stats.expanding_corr(df, 'c').values
        assert_b = np.array([
            [np.nan, np.nan, np.nan],
            [1.0, -1.0, 1.0],
            [0.9078412990032035, -0.9078412990032034, 1.0],
            [0.824615943275715, -0.824615943275715, 1.0],
            [0.7597711308538598, -0.7597711308538598, 1.0],
            [0.7074867359704088, -0.7074867359704086, 1.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_expanding_corr_backwards(self):
        assert_a = stats.expanding_corr(df, 'c', backwards=True).values
        assert_b = np.array([
            [0.7074867359704088, -0.7074867359704086, 1.0],
            [0.7597208512063609, -0.7597208512063609, 1.0],
            [0.824140716620699, -0.824140716620699, 1.0],
            [0.9041944301794651, -0.9041944301794651, 1.0],
            [1.0, -1.0, 1.0],
            [np.nan, np.nan, np.nan]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_resampling_corr(self):
        assert_a = stats.resampling_corr(df, 'c', pd.Timedelta(minutes=1)).values
        assert_b = np.array([
            [0.9078412990032035, -0.9078412990032035, 0.9999999999999999],
            [0.9041944301794651, -0.9041944301794651, 1.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    # Percentiles

    def test_percentiles(self):
        assert_a = stats.percentiles(df, 0, 100, 10).values
        assert_b = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 5.0],
            [1.0, 1.0, 10.0],
            [1.5, 1.5, 55.0],
            [2.0, 2.0, 100.0],
            [2.5, 2.5, 550.0],
            [3.0, 3.0, 1000.0],
            [3.5, 3.5, 5500.0],
            [4.0, 4.0, 10000.0],
            [4.5, 4.5, 55000.0],
            [5.0, 5.0, 100000.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    # Max drawdown

    def test_mdd(self):
        assert_a = stats.mdd(dd_df).values
        assert_b = np.array([
            [0.0, 0.0],
            [0.0, 0.33333333333333337],
            [0.0, 0.6666666666666667],
            [0.33333333333333337, 0.33333333333333337],
            [0.6666666666666667, 0.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_rolling_mdd(self):
        assert_a = stats.rolling_mdd(dd_df, window=3).values
        assert_b = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan],
            [0.0, 0.6666666666666667],
            [0.33333333333333337, 0.0],
            [0.6666666666666667, 0.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_resampling_mdd(self):
        assert_a = stats.resampling_mdd(dd_df, pd.Timedelta(minutes=1)).values
        assert_b = np.array([
            [0.0, 0.0],
            [0.0, 0.33333333333333337],
            [0.0, 0.6666666666666667],
            [0.33333333333333337, 0.33333333333333337],
            [0.6666666666666667, 0.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_dd_info(self):
        assert_a_a = stats.dd_info(dd_df)['a']
        assert_b_a = pd.DataFrame([[
            pd.Timestamp('2018-01-01 01:01:03'),
            pd.Timestamp('2018-01-01 01:01:05'),
            None,
            pd.Timedelta('0 days 00:00:02'),
            None,
            66.66666666666667,
            None
        ]], columns=assert_a_a.columns)
        pd.testing.assert_frame_equal(assert_a_a, assert_b_a)
        assert_a_b = stats.dd_info(dd_df)['b']
        assert_b_b = pd.DataFrame([[
            pd.Timestamp('2018-01-01 01:01:01'),
            pd.Timestamp('2018-01-01 01:01:03'),
            pd.Timestamp('2018-01-01 01:01:05'),
            pd.Timedelta('0 days 00:00:02'),
            pd.Timedelta('0 days 00:00:02'),
            66.66666666666667,
            200.0
        ]], columns=assert_a_b.columns)
        pd.testing.assert_frame_equal(assert_a_b, assert_b_b)

if __name__ == '__main__':
    unittest.main()