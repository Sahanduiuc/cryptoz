import unittest

from cryptoz import utils

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.DataFrame(
    {
        'a': [0, 1, 2, 3, 4], 
        'b': [4, 3, 2, 1, 0], 
        'c': [0, -1, -2, -3, -4]
    }, index=[
    datetime(2018, 1, 1, 1, 1, 1),
    datetime(2018, 1, 1, 1, 1, 2),
    datetime(2018, 1, 1, 1, 1, 3),
    datetime(2018, 1, 1, 1, 2, 1),
    datetime(2018, 1, 1, 1, 2, 2)
])


class TestUtils(unittest.TestCase):

    def test_select_window(self):
        assert_a = utils.select_window(df, pd.Timedelta(minutes=1)).values
        assert_b = np.array([[2, 2, -2], [3, 1, -3], [4, 0, -4]])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_describe(self):
        assert_a = utils.describe(df).values
        assert_b = np.array([
            [5.0, 2.0, 1.5811388300841898, 0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 2.0, 1.5811388300841898, 0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, -2.0, 1.5811388300841898, -4.0, -3.0, -2.0, -1.0, 0.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    # Windows

    def test_apply(self):
        assert_a = utils.apply(df, lambda x: x + np.sum(x)).values
        assert_b = np.array([[10, 14, -10], [11, 13, -11], [12, 12, -12], [13, 11, -13], [14, 10, -14]])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_rolling_apply(self):
        assert_a = utils.rolling_apply(df, lambda x: (x + np.sum(x))[-1], window=3).values
        assert_b = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [5.0, 11.0, -5.0],
            [9.0, 7.0, -9.0],
            [13.0, 3.0, -13.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_rolling_apply_backwards(self):
        assert_a = utils.rolling_apply(df, lambda x: (x + np.sum(x))[-1], backwards=True, window=3).values
        assert_b = np.array([
            [3.0, 13.0, -3.0],
            [7.0, 9.0, -7.0],
            [11.0, 5.0, -11.0],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_expanding_apply(self):
        assert_a = utils.expanding_apply(df, lambda x: (x + np.sum(x))[-1]).values
        assert_b = np.array([
            [0.0, 8.0, 0.0],
            [2.0, 10.0, -2.0],
            [5.0, 11.0, -5.0],
            [9.0, 11.0, -9.0],
            [14.0, 10.0, -14.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_expanding_apply_backwards(self):
        assert_a = utils.expanding_apply(df, lambda x: (x + np.sum(x))[-1], backwards=True).values
        assert_b = np.array([
            [10.0, 14.0, -10.0],
            [11.0, 9.0, -11.0],
            [11.0, 5.0, -11.0],
            [10.0, 2.0, -10.0],
            [8.0, 0.0, -8.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_resampling_apply(self):
        assert_a = utils.resampling_apply(df, lambda x: x + np.sum(x), pd.Timedelta(minutes=1)).values
        assert_b = np.array([[3, 13, -3], [4, 12, -4], [5, 11, -5], [10, 2, -10], [11, 1, -11]])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    # Combinations

    def test_product(self):
        assert_a = utils.product(df.columns)
        assert_b = [
            ('a', 'a'),
            ('a', 'b'),
            ('a', 'c'),
            ('b', 'a'),
            ('b', 'b'),
            ('b', 'c'),
            ('c', 'a'),
            ('c', 'b'),
            ('c', 'c')
        ]
        assert(assert_a == assert_b)

    def test_combine(self):
        assert_a = utils.combine(df.columns)
        assert_b = [('a', 'b'), ('a', 'c'), ('b', 'c')]
        assert(assert_a == assert_b)

    def test_combine(self):
        assert_a = utils.combine_rep(df.columns)
        assert_b = [('a', 'a'), ('a', 'b'), ('a', 'c'), ('b', 'b'), ('b', 'c'), ('c', 'c')]
        assert(assert_a == assert_b)

    def test_pairwise_apply(self):
        assert_a = utils.pairwise_apply(df, utils.combine_rep, lambda x, y: x + y).values
        assert_b = np.array([
            [0, 4, 0, 8, 4, 0],
            [2, 4, 0, 6, 2, -2],
            [4, 4, 0, 4, 0, -4],
            [6, 4, 0, 2, -2, -6],
            [8, 4, 0, 0, -4, -8]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    # Normalization

    def test_normalize(self):
        assert_a = utils.normalize(df, 'minmax').values
        assert_b = np.array([
            [0.0, 1.0, 1.0],
            [0.25, 0.75, 0.75],
            [0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25],
            [1.0, 0.0, 0.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_rolling_normalize(self):
        assert_a = utils.rolling_normalize(df, 'minmax', window=3).values
        assert_b = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_expanding_normalize(self):
        assert_a = utils.expanding_normalize(df, 'minmax').values
        assert_b = np.array([
            [np.nan, np.nan, np.nan],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_resampling_normalize(self):
        assert_a = utils.resampling_normalize(df, 'minmax', pd.Timedelta(minutes=1)).values
        assert_b = np.array([
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    # Rescaling

    def test_rescale(self):
        assert_a = utils.rescale(df, [0, 1]).values
        assert_b = np.array([
            [0.0, 1.0, 1.0],
            [0.25, 0.75, 0.75],
            [0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25],
            [1.0, 0.0, 0.0]
        ])
        # Must be the same as minmax normalization
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_rescale_from_range(self):
        assert_a = utils.rescale(df, [0, 1], from_range=[0, 10]).values
        assert_b = np.array([
            [0.0, 0.4, 0.0],
            [0.1, 0.3, -0.1],
            [0.2, 0.2, -0.2],
            [0.3, 0.1, -0.3],
            [0.4, 0.0, -0.4]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_rolling_rescale(self):
        assert_a = utils.rolling_rescale(df, [0, 1], window=3).values
        assert_b = np.array([
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_expanding_rescale(self):
        assert_a = utils.expanding_rescale(df, [0, 1]).values
        assert_b = np.array([
            [np.nan, np.nan, np.nan],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_resampling_rescale(self):
        assert_a = utils.resampling_rescale(df, [0, 1], pd.Timedelta(minutes=1)).values
        assert_b = np.array([
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_dynamic_rescale(self):
        assert_a = utils.dynamic_rescale(df, [df*0, df*0+10], [0, 1]).values
        assert_b = np.array([
            [0.0, 0.4, 0.0],
            [0.1, 0.3, -0.1],
            [0.2, 0.2, -0.2],
            [0.3, 0.1, -0.3],
            [0.4, 0.0, -0.4]
        ])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_reverse_scale(self):
        assert_a = utils.reverse_scale(df).values
        assert_b = np.array([[4, 0, -4], [3, 1, -3], [2, 2, -2], [1, 3, -1], [0, 4, 0]])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

if __name__ == '__main__':
    unittest.main()