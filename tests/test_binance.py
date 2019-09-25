import unittest

from cryptoz import binance

import pandas as pd
import numpy as np
import pickle
from cryptoz.binance import BinanceHelper
from binance import enums

class BinanceClientMock:
    # A simple class mock for https://github.com/sammchardy/python-binance/binance/client.py
    def get_all_tickers(self):
        with open("tests/mock_data/get_all_tickers.pkl", "rb") as f:
            return pickle.load(f)

    def get_klines(self, symbol, **params): 
        with open(f"tests/mock_data/get_klines_{symbol}.pkl", "rb") as f:
            return pickle.load(f)

    def get_order_book(self, symbol, **params): 
        with open(f"tests/mock_data/get_order_book_{symbol}.pkl", "rb") as f:
            return pickle.load(f)
        
client_mock = BinanceClientMock()
binance = BinanceHelper(client_mock)
interval = enums.KLINE_INTERVAL_1HOUR
ticker = binance.get_ticker()

class TestBinance(unittest.TestCase):

    # Symbols and pairs

    def test_symbol_into_pair(self):
        assert(binance.symbol_into_pair("BTCUSDT") == ('BTC', 'USDT'))

    def test_pair_into_symbol(self):
        assert(binance.pair_into_symbol(('BTC', 'USDT')) == "BTCUSDT")

    def test_get_pairs(self):
        assert(len(binance.get_pairs()) == 518)

    # Ticker

    def test_price_BTC_USDT(self):
        assert(binance.calculate_pair_price(('BTC', 'USDT'), ticker) == 8643.24)

    def test_price_GAS_USDT(self):
        # GAS/USDT = (GAS/BTC) * (BTC/USDT)
        assert(binance.calculate_pair_price(('GAS', 'USDT'), ticker) == 1.07176176)

    def test_price_GAS_LSK(self):
        # GAS/LSK = (GAS/BTC) / (LSK/BTC)
        assert(binance.calculate_pair_price(('GAS', 'LSK'), ticker) == 1.238761238761239)

    def test_price_USDT_BTC(self):
        # USDT/BTC = 1 / (BTC/USDT)
        assert(binance.calculate_pair_price(('USDT', 'BTC'), ticker) == 0.0001156973542329034)

    # OHLCV

    def test_ohlcv_BTC_USDT(self):
        assert_a = binance.get_pair_ohlcv(('BTC', 'USDT'), interval).head().values.tolist()
        assert_b = np.array([[
            10578.58, 
            10597.18, 
            10531.82, 
            10560.99, 
            9184505.29453171, 
            10563.33],
            [10559.21,
            10612.97,
            10558.48,
            10571.26,
            8546932.17347709,
            10580.903333333334],
            [10572.62, 10592.19, 10531.02, 10568.0, 9358340.95451738, 10563.736666666666],
            [10569.1,
            10578.86,
            10451.13,
            10502.99,
            20973398.14048234,
            10510.993333333332],
            [10504.0,
            10538.36,
            10485.51,
            10498.27,
            11470686.80424993,
            10507.380000000001]])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_ohlcv_GAS_USDT(self):
        # GAS/USDT = (GAS/BTC) * (BTC/USDT)
        assert_a = binance.get_pair_ohlcv(('GAS', 'USDT'), interval).head().values.tolist()
        assert_b = np.array([[
            1.4281083, 
            1.43661288, 
            1.42604955, 
            1.43629464, 
            5467.9255570575, 
            1.43309177],
            [1.4360525599999998,
            1.4390028533333334,
            1.4284219500000002,
            1.4271201,
            193.91536891773333,
            1.4319489177777778],
            [1.4273037000000002,
            1.4366681866666666,
            1.42610445,
            1.437248,
            2338.6031923876667,
            1.433146941111111],
            [1.4373976,
            1.4400060866666664,
            1.4189840999999999,
            1.41790365,
            1999.9467775305995,
            1.4259914288888886],
            [1.41804,
            1.42900368,
            1.4079889200000002,
            1.42776472,
            4003.3030588746005,
            1.4219987600000001]])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_ohlcv_GAS_LSK(self):
        # GAS/LSK = (GAS/BTC) / (LSK/BTC)
        assert_a = binance.get_pair_ohlcv(('GAS', 'LSK'), interval).head().values.tolist()
        assert_b = np.array([[
            1.2652296157450795,
            1.2746016869728212,
            1.2652296157450797,
            1.2734082397003745,
            4851.291002811621,
            1.2714776632302407],
            [1.2710280373831775,
            1.2726138490330632,
            1.2632563942607613,
            1.2605042016806722,
            171.4933250155958,
            1.2663755458515285],
            [1.262862488306829,
            1.275398562050641,
            1.2660206314473275,
            1.2769953051643192,
            2076.0890903407317,
            1.2722725851828698],
            [1.274601686972821,
            1.2831720262254134,
            1.2644395878863564,
            1.2605042016806722,
            1782.1284108648138,
            1.2706837339993753],
            [1.2605042016806722,
            1.2734082397003745,
            1.254681647940075,
            1.2698412698412698,
            3567.4079588014984,
            1.2671660424469413]])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_ohlcv_USDT_BTC(self):
        # USDT/BTC = 1 / (BTC/USDT)
        assert_a = binance.get_pair_ohlcv(('USDT', 'BTC'), interval).head().values.tolist()
        assert_b = np.array([[
            9.45306458900911e-05,
            9.495035046174356e-05,
            9.436472721988302e-05,
            9.468809268828017e-05,
            869.4706398959145,
            9.466711728214493e-05],
            [9.470405456468809e-05,
            9.471060228366205e-05,
            9.422433117214126e-05,
            9.459610301894003e-05,
            807.7696113668704,
            9.450988904223993e-05],
            [9.458393472951831e-05,
            9.495756346488753e-05,
            9.440918261473784e-05,
            9.462528387585163e-05,
            885.8930556312662,
            9.466347293145323e-05],
            [9.46154355621576e-05,
            9.56834332746794e-05,
            9.452814386427271e-05,
            9.521098277728532e-05,
            1995.3773611452843,
            9.513848675259999e-05],
            [9.52018278750952e-05,
            9.536970543159083e-05,
            9.48914252312504e-05,
            9.525378943387815e-05,
            1091.6790678789507,
            9.517120347793645e-05]])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    # Depth

    def test_depth_BTC_USDT(self):
        assert_a = binance.get_pair_depth(('BTC', 'USDT')).head().values.tolist()
        assert_b = np.array([
            610158.4841830102,
            609260.7729523102,
            600621.9629523101,
            596302.53795231,
            590306.2498477501])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_depth_GAS_USDT(self):
        # GAS/USDT = (GAS/BTC) * (BTC/USDT)
        assert_a = binance.get_pair_depth(('GAS', 'USDT')).head().values.tolist()
        assert_b = np.array([
            63976.34737696498,
            63958.71516736498,
            63906.41526785458,
            63871.84230785458,
            63854.55582785459])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_depth_GAS_LSK(self):
        # GAS/LSK = (GAS/BTC) / (LSK/BTC)
        assert_a = binance.get_pair_depth(('GAS', 'LSK')).head().values.tolist()
        assert_b = np.array([
            73944.99625374623,
            73924.61663336662,
            73864.16748251747,
            73824.20744255742,
            73804.22742257742])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

    def test_depth_USDT_BTC(self):
        # USDT/BTC = 1 / (BTC/USDT)
        assert_a = binance.get_pair_depth(('USDT', 'BTC')).head().values.tolist()
        assert_b = np.array([
            70.59372228273311,
            70.48985946847596,
            69.49037200775521,
            68.99062596344774,
            68.29687129453193])
        assert(np.allclose(assert_a, assert_b, equal_nan=True))

