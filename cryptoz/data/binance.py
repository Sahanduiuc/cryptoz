import re
from datetime import datetime

import pandas as pd


def ts_to_dt(ts):
    return datetime.utcfromtimestamp(ts / 1000)


def _symbols(client):
    """Extract symbols from ticker"""
    return set(map(lambda d: d['symbol'], client.get_all_tickers()))


def _price(client, symbol):
    """Extract the most recent price"""
    return float(client.get_recent_trades(symbol=symbol, limit=1)[0]['price'])


def _chartdata(client, **params):
    """Load OHLC data on a single symbol"""
    candles = client.get_klines(**params)
    columns = ['date', 'O', 'H', 'L', 'C', '_', '_', 'V', '_', '_', '_', '_']
    chart_df = pd.DataFrame(candles, columns=columns)
    chart_df.set_index('date', drop=True, inplace=True)
    chart_df.index = [ts_to_dt(i) for i in chart_df.index]
    chart_df.fillna(method='ffill', inplace=True)  # fill gaps forwards
    chart_df.fillna(method='bfill', inplace=True)  # fill gaps backwards
    chart_df = chart_df.astype(float)
    return chart_df[['O', 'H', 'L', 'C', 'V']]


def _convert(ohlcA, ohlcB):
    """Example: get XRP/USDT from XRP/BTC and BTC/USDT"""
    ohlc = ohlcA.copy()
    ohlc['O'] *= ohlcB['O']
    ohlc['H'] *= ohlcB['H'] # incomplete info -> inaccurate -> choose small intervals
    ohlc['L'] *= ohlcB['L'] # inaccurate
    ohlc['C'] *= ohlcB['C']
    ohlc['V'] *= ohlcB['C'] # arbitrary
    return ohlc


def chartdata(client, symbols, **params):
    all_symbols = _symbols(client)
    get_chartdata = lambda symbol: _chartdata(client, symbol=symbol, **params)

    if isinstance(symbols, str):
        regex = re.compile(symbols)
        symbols = list(filter(regex.search, all_symbols))
        return dict(zip(symbols, map(get_chartdata, symbols)))

    print("%d symbols:" % len(symbols))

    # Symbols must be in format BASE/QUOTE to be split easier
    # If pair not supported by Binance -> convert using BTC in first place
    BTCUSDT = None
    ohlc = {}
    for symbol in symbols:
        base, quote = symbol.split('/')
        joined = ''.join([base, quote])
        print("%s.. " % joined, end='')
        if joined in all_symbols:
            ohlc[joined] = get_chartdata(joined)
        else:
            if quote == 'USDT':
                # translate to USDT
                if BTCUSDT is None:
                    BTCUSDT = get_chartdata('BTCUSDT')
                ohlc[joined] = _convert(get_chartdata(''.join([base, 'BTC'])), BTCUSDT)
            else:
                raise Exception("Symbol %s not found." % symbol)
        print("done")
    return ohlc


def _pack_orderbook(orderbook):
    """Transform orderbook into series"""
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


def orderbooks(client, symbols, **params):
    """Load and pack orderbooks"""
    all_symbols = _symbols(client)
    process_orderbook = lambda symbol: _pack_orderbook(client.get_order_book(symbol=symbol, **params))

    if isinstance(symbols, str):
        regex = re.compile(symbols)
        symbols = list(filter(regex.search, all_symbols))
        return dict(zip(symbols, map(process_orderbook, symbols)))

    print("%d symbols:" % len(symbols))

    BTCUSDT = None
    orderbooks = {}
    for symbol in symbols:
        base, quote = symbol.split('/')
        joined = ''.join([base, quote])
        print("%s.. " % joined, end='')
        if joined in all_symbols:
            orderbooks[joined] = process_orderbook(joined)
        else:
            if quote == 'USDT':
                # translate to USDT
                if BTCUSDT is None:
                    BTCUSDT = _price(client, 'BTCUSDT')
                orderbook = process_orderbook(''.join([base, 'BTC'])) * BTCUSDT
                orderbook.index *= BTCUSDT
                orderbooks[joined] = orderbook
            else:
                raise Exception("Symbol %s not found." % symbol)
        print("done")
    return orderbooks
