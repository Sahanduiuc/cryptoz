from datetime import datetime, timedelta

import pandas as pd
import pytz

from cryptoz.exchanges import Exchange


class Poloniex(Exchange):
    PERIOD_5MINUTE = 300
    PERIOD_15MINUTE = 900
    PERIOD_30MINUTE = 1800
    PERIOD_2HOUR = 7200
    PERIOD_4HOUR = 14400
    PERIOD_1DAY = 86400

    def __init__(self, client):
        Exchange.__init__(self, client)

    @staticmethod
    def now():
        return Poloniex._dt_to_ts(pytz.utc.localize(datetime.utcnow()))

    @staticmethod
    def ago(**kwargs):
        return Poloniex.now() - timedelta(**kwargs).total_seconds()

    @staticmethod
    def _dt_to_ts(date):
        return int(date.timestamp())

    @staticmethod
    def _to_intern_pair(pair):
        """BTCUSDT to BTC/USDT"""
        if '/' in pair:
            return pair
        supported_quotes = ['USDT', 'BTC', 'ETH', 'XMR']
        quote, base = pair.split('_')
        if quote in supported_quotes:
            return base + '/' + quote
        return None

    @staticmethod
    def _to_exchange_pair(pair):
        """BTC/USDT to BTCUSDT"""
        if '/' not in pair:
            return pair
        base, quote = pair.split('/')
        return quote + '_' + base

    def get_pairs(self):
        pairs = set(map(lambda s: self._to_intern_pair(str(s).upper()), self.client.returnTicker().keys()))
        return list(filter(lambda s: s is not None, pairs))

    def get_price(self, pair):
        pair = self._to_exchange_pair(pair)
        return float(self.client.returnTicker()[pair]['last'])

    def _get_ohlc(self, pair, *args, **kwargs):
        pair = self._to_exchange_pair(pair)
        chart_data = self.client.returnChartData(pair, *args, **kwargs)

        df = pd.DataFrame(chart_data)
        df.set_index('date', drop=True, inplace=True)
        df.index = pd.to_datetime(df.index, unit='s')
        df.fillna(method='ffill', inplace=True)  # fill gaps forwards
        df.fillna(method='bfill', inplace=True)  # fill gaps backwards
        df = df.astype(float)
        df.rename(columns={'open': 'O', 'high': 'H', 'low': 'L', 'close': 'C', 'volume': 'V'}, inplace=True)
        df = df[['O', 'H', 'L', 'C', 'V']]
        df['M'] = (df['L'] + df['H'] + df['C']) / 3
        df = df.iloc[1:] # first entry can be dirty
        return df

    def get_ohlc(self, pairs, *args, **kwargs):
        load_func = lambda pair: self._get_ohlc(pair, *args, **kwargs)
        return Exchange._load_pairs(self, pairs, Exchange._convert_ohlc, load_func)

    def _get_orderbook(self, pair, **kwargs):
        pair = self._to_exchange_pair(pair)
        orderbook = self.client.returnOrderBook(currencyPair=pair, **kwargs)

        rates, amounts = zip(*orderbook['bids'])
        cum_bids = pd.Series(amounts, index=rates, dtype=float)
        cum_bids.index = cum_bids.index.astype(float)
        cum_bids = cum_bids.sort_index(ascending=False).cumsum().sort_index()
        cum_bids *= cum_bids.index

        rates, amounts = zip(*orderbook['asks'])
        cum_asks = pd.Series(amounts, index=rates, dtype=float)
        cum_asks.index = cum_asks.index.astype(float)
        cum_asks = -cum_asks.sort_index().cumsum()
        cum_asks *= cum_asks.index

        return cum_bids.append(cum_asks).sort_index()

    def get_orderbooks(self, pairs, **kwargs):
        load_func = lambda pair: self._get_orderbook(pair, **kwargs)
        return Exchange._load_pairs(self, pairs, Exchange._convert_orderbook, load_func, cross_func=self.get_price)
