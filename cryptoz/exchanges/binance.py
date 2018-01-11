from cryptoz.data import Exchange
from datetime import datetime

import pandas as pd


class Binance(Exchange):
    def __init__(self, client):
        super.__init__(client)

    def _ts_to_dt(self, ts):
        return datetime.utcfromtimestamp(ts / 1000)

    def _to_intern_pair(self, pair):
        """BTCUSDT to BTC/USDT"""
        if '/' in pair:
            return pair
        supported_quotes = ['USDT', 'BTC', 'ETH', 'BNB']
        for quote in supported_quotes:
            if pair[-len(quote):] == quote:
                return pair[:-len(quote)] + '/' + quote
        return None

    def _to_exchange_pair(self, pair):
        """BTC/USDT to BTCUSDT"""
        if '/' not in pair:
            return pair
        return ''.join(pair.split('/'))

    def get_pairs(self):
        pairs = set(map(lambda d: self._to_intern_pair(d['symbol']), self.client.get_all_tickers()))
        return list(filter(lambda s: s is not None, pairs))

    def get_price(self, pair):
        pair = self._to_exchange_pair(pair)
        return float(self.client.get_recent_trades(symbol=pair, limit=1)[0]['price'])

    def _get_ohlc(self, **params):
        params['symbol'] = self._to_exchange_pair(params['symbol'])
        """Load OHLC data on a single pair"""
        candles = self.client.get_klines(**params)
        columns = ['date', 'O', 'H', 'L', 'C', '_', '_', 'V', '_', '_', '_', '_']
        chart_df = pd.DataFrame(candles, columns=columns)
        chart_df.set_index('date', drop=True, inplace=True)
        chart_df.index = [self._ts_to_dt(i) for i in chart_df.index]
        chart_df.fillna(method='ffill', inplace=True)  # fill gaps forwards
        chart_df.fillna(method='bfill', inplace=True)  # fill gaps backwards
        chart_df = chart_df.astype(float)
        return chart_df[['O', 'H', 'L', 'C', 'V']].iloc[1:]  # first entry can be dirty

    def _convert_ohlc(self, ohlc, cross_ohlc, divide=True):
        if divide:
            cross_ohlc = 1 / cross_ohlc.copy()
        ohlc = ohlc.copy()
        ohlc['O'] *= cross_ohlc['O']
        middle = (cross_ohlc['L'] + cross_ohlc['H'] + cross_ohlc['C']) / 3  # middle
        ohlc['H'] *= middle
        ohlc['L'] *= middle
        ohlc['C'] *= cross_ohlc['C']
        ohlc['V'] *= middle
        return ohlc

    def get_ohlc(self, pairs, **params):
        supported = self.get_pairs()
        load_func = lambda pair: self._get_ohlc(symbol=pair, **params)
        return self._load_pairs(pairs, supported, self._convert_ohlc, load_func)

    def _get_orderbook(self, pair, **params):
        pair = self._to_exchange_pair(pair)
        orderbook = self.client.get_order_book(symbol=pair, **params)

        rates, amounts, _ = zip(*orderbook['bids'])
        cum_bids = pd.Series(amounts, index=rates, dtype=float)
        cum_bids.index = cum_bids.index.astype(float)
        cum_bids = cum_bids.sort_index(ascending=False).cumsum().sort_index()
        cum_bids *= cum_bids.index

        rates, amounts, _ = zip(*orderbook['asks'])
        cum_asks = pd.Series(amounts, index=rates, dtype=float)
        cum_asks.index = cum_asks.index.astype(float)
        cum_asks = -cum_asks.sort_index().cumsum()
        cum_asks *= cum_asks.index

        return cum_bids.append(cum_asks).sort_index()

    def _convert_orderbook(self, orderbook, cross_price, divide=True):
        if divide:
            cross_price = 1 / cross_price
        orderbook = orderbook.copy()
        orderbook *= cross_price
        orderbook.index *= cross_price
        return orderbook

    def get_orderbooks(self, pairs, **params):
        supported = self.get_pairs()
        load_func = lambda pair: self._get_orderbook(pair, **params)
        cross_func = lambda pair: self.get_price(pair)
        return self.load_pairs(pairs, supported, self._convert_orderbook, load_func, cross_func=cross_func)
