import re
from datetime import datetime, timedelta
from enum import Enum

import pandas as pd
import pytz
from poloniex import Poloniex

polo = Poloniex()


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


def dt_to_ts(date):
    return int(date.timestamp())


def ticker_pairs():
    """Return all pairs from ticker"""
    ticker = polo.returnTicker()
    pairs = set(map(lambda x: str(x).upper(), ticker.keys()))
    return pairs


def _chartdata(pair, from_dt, to_dt, period=None):
    """Load OHLC data on a single pair"""
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
    chart_data = polo.returnChartData(pair, period=period, start=dt_to_ts(from_dt), end=dt_to_ts(to_dt))
    chart_df = pd.DataFrame(chart_data)
    chart_df.set_index('date', drop=True, inplace=True)
    chart_df.index = pd.to_datetime(chart_df.index, unit='s')
    chart_df.fillna(method='ffill', inplace=True)  # fill gaps forwards
    chart_df.fillna(method='bfill', inplace=True)  # fill gaps backwards
    chart_df = chart_df.astype(float)
    chart_df.rename(columns={'open': 'O', 'high': 'H', 'low': 'L', 'close': 'C', 'volume': 'V'}, inplace=True)
    return chart_df[['O', 'H', 'L', 'C', 'V']]


def chartdata(pairs, *args, **kwargs):
    if isinstance(pairs, str):
        regex = re.compile(pairs)
        pairs = list(filter(regex.search, ticker_pairs()))
    return {pair: _chartdata(pair, *args, **kwargs) for pair in pairs}


def _pack_orderbook(orderbook):
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


def orderbooks(pairs, depth=100):
    """Load and pack orderbooks"""
    if isinstance(pairs, str):
        regex = re.compile(pairs)
        pairs = list(filter(regex.search, ticker_pairs()))
    orderbooks = polo.returnOrderBook(currencyPair='all', depth=depth)

    return {pair: _pack_orderbook(orderbooks[pair]) for pair in pairs}
