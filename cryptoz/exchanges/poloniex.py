import re
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd
import pytz


class Period(Enum):
    M5 = 300
    M15 = 900
    M30 = 1800
    H2 = 7200
    H4 = 14400
    D1 = 86400


def now_dt():
    return pytz.utc.localize(datetime.utcnow())


def ago_dt(**kwargs):
    return now_dt() - timedelta(**kwargs)


def _dt_to_ts(date):
    return int(date.timestamp())


def _to_intern_pair(pair):
    """BTCUSDT to BTC/USDT"""
    if '/' in pair:
        return pair
    supported_quotes = ['USDT', 'BTC', 'ETH', 'XMR']
    quote, base = pair.split('_')
    if quote in supported_quotes:
        return base + '/' + quote
    return None


def _to_exchange_pair(pair):
    """BTC/USDT to BTCUSDT"""
    if '/' not in pair:
        return pair
    base, quote = pair.split('/')
    return quote + '_' + base


def get_pairs(client):
    pairs = set(map(lambda s: _to_intern_pair(str(s).upper()), client.returnTicker().keys()))
    return list(filter(lambda s: s is not None, pairs))


def _get_ohlc(client, pair, from_dt, to_dt, period=None):
    if period is None:
        periods = sorted(list(map(lambda p: p.value, Period)), reverse=True)
        for p in periods:
            if (to_dt - from_dt).total_seconds() >= p:
                period = p
                break
        if period is None:
            raise Exception("Period too narrow")

    if isinstance(period, Period):
        period = period.value
    chart_data = client.returnChartData(pair, period=period, start=_dt_to_ts(from_dt), end=_dt_to_ts(to_dt))
    chart_df = pd.DataFrame(chart_data)
    chart_df.set_index('date', drop=True, inplace=True)
    chart_df.index = pd.to_datetime(chart_df.index, unit='s')
    chart_df.fillna(method='ffill', inplace=True)  # fill gaps forwards
    chart_df.fillna(method='bfill', inplace=True)  # fill gaps backwards
    chart_df = chart_df.astype(float)
    chart_df.rename(columns={'open': 'O', 'high': 'H', 'low': 'L', 'close': 'C', 'volume': 'V'}, inplace=True)
    return chart_df[['O', 'H', 'L', 'C', 'V']]


def get_ohlc(client, pairs, *args, **kwargs):
    if isinstance(pairs, str):
        regex = re.compile(pairs)
        pairs = list(filter(regex.search, get_pairs()))
    return {pair: _get_ohlc(client, pair, *args, **kwargs) for pair in pairs}


def _process_orderbook(orderbook):
    """Transform orderbook into series"""
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


def get_orderbooks(client, pairs, depth=100):
    """Load and process orderbooks"""
    if isinstance(pairs, str):
        regex = re.compile(pairs)
        pairs = list(filter(regex.search, get_pairs()))
    orderbooks = client.returnOrderBook(currencyPair='all', depth=depth)

    return {pair: _process_orderbook(orderbooks[pair]) for pair in pairs}
