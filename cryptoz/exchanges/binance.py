from datetime import datetime

import pandas as pd

from cryptoz.exchanges import Exchange


class Binance(Exchange):
    def __init__(self, client):
        Exchange.__init__(self, client)

    @staticmethod
    def _ts_to_dt(ts):
        return datetime.utcfromtimestamp(ts / 1000)

    @staticmethod
    def _to_intern_pair(pair):
        """BTCUSDT to BTC/USDT"""
        if '/' in pair:
            return pair
        supported_quotes = ['USDT', 'BTC', 'ETH', 'BNB']
        for quote in supported_quotes:
            if pair[-len(quote):] == quote:
                return pair[:-len(quote)] + '/' + quote
        return None

    @staticmethod
    def _to_exchange_pair(pair):
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

    def _get_ohlc(self, **kwargs):
        kwargs['symbol'] = self._to_exchange_pair(kwargs['symbol'])
        """Load OHLC data on a single pair"""
        candles = self.client.get_klines(**kwargs)
        columns = ['date', 'O', 'H', 'L', 'C', '_', '_', 'V', '_', '_', '_', '_']

        df = pd.DataFrame(candles, columns=columns)
        df.set_index('date', drop=True, inplace=True)
        df.index = [self._ts_to_dt(i) for i in df.index]
        df.fillna(method='ffill', inplace=True)  # fill gaps forwards
        df.fillna(method='bfill', inplace=True)  # fill gaps backwards
        df = df.astype(float)
        df = df[['O', 'H', 'L', 'C', 'V']]
        df['M'] = (df['L'] + df['H'] + df['C']) / 3
        df = df.iloc[1:] # first entry can be dirty
        return df

    def get_ohlc(self, pairs, **kwargs):
        load_func = lambda pair: self._get_ohlc(symbol=pair, **kwargs)
        return Exchange._load_pairs(self, pairs, Exchange._convert_ohlc, load_func)

    def _get_orderbook(self, pair, **kwargs):
        pair = self._to_exchange_pair(pair)
        orderbook = self.client.get_order_book(symbol=pair, **kwargs)

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

    def get_orderbooks(self, pairs, **kwargs):
        load_func = lambda pair: self._get_orderbook(pair, **kwargs)
        return Exchange._load_pairs(self, pairs, Exchange._convert_orderbook, load_func, cross_func=self.get_price)
