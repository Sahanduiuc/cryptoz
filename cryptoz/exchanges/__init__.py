import re

import pandas as pd


class Exchange(object):
    def __init__(self, client):
        self.client = client

    def get_pairs(self):
        pass

    def get_ticker(self):
        pass

    def _get_ohlc(self, pair, *args, **kwargs):
        pass

    def get_ohlc(self, pairs, *args, **kwargs):
        pass

    def _get_orderbook(self, pair, *args, **kwargs):
        pass

    def get_orderbooks(self, pairs, *args, **kwargs):
        pass

    def _get_balances(self):
        pass

    def get_balances(self, hide_small_assets=True):
        pass

    @staticmethod
    def _get_price(base, quote, ticker):
        if base == quote:
            return 1.0
        elif base + '/' + quote in ticker:
            return ticker[base + '/' + quote]
        elif base + '/BTC' in ticker and 'BTC/' + quote in ticker:
            return ticker[base + '/BTC'] * ticker['BTC/' + quote]
        elif base + '/BTC' in ticker and quote + '/BTC' in ticker:
            return ticker[base + '/BTC'] / ticker[quote + '/BTC']
        elif quote + '/' + base in ticker:
            return 1 / ticker[quote + '/' + base]
        return None

    @staticmethod
    def _convert_ohlc(ohlc, cross_ohlc, divide=False):
        if divide:
            cross_ohlc = 1 / cross_ohlc
        ohlc = ohlc.copy()
        ohlc['O'] *= cross_ohlc['O']
        middle = (cross_ohlc['L'] + cross_ohlc['H'] + cross_ohlc['C']) / 3  # TP
        ohlc['H'] *= middle
        ohlc['L'] *= middle
        ohlc['C'] *= cross_ohlc['C']
        ohlc['V'] *= middle
        return ohlc

    @staticmethod
    def _convert_orderbook(orderbook, cross_price, divide=False):
        if divide:
            cross_price = 1 / cross_price
        orderbook = orderbook.copy()
        orderbook *= cross_price
        orderbook.index *= cross_price
        return orderbook

    @staticmethod
    def _get_crosspairs(pair, pairs):
        # returns: pair1, pair2, divide
        base, quote = pair.split('/')
        if base + '/BTC' in pairs and 'BTC/' + quote in pairs:
            # XMR/USDT: XMR/BTC * BTC/USDT
            return base + '/BTC', 'BTC/' + quote, False
        elif base + '/BTC' in pairs and quote + '/BTC' in pairs:
            # XMR/DASH: XMR/BTC / DASH/BTC
            return base + '/BTC', quote + '/BTC', True
        else:
            raise Exception("Pair cannot be converted.")

    @staticmethod
    def _load_and_convert_balances(child, hide_small_assets=True):
        balances = child._get_balances()
        ticker = child.get_ticker()
        df = pd.DataFrame(list(balances.values()), index=list(balances.keys()), columns=['quantity'])
        df['priceBTC'] = list(map(lambda base: Exchange._get_price(base, 'BTC', ticker), df.index))
        df['priceUSDT'] = list(map(lambda base: Exchange._get_price(base, 'USDT', ticker), df.index))
        df['BTC'] = df['quantity'] * df['priceBTC']
        df['USDT'] = df['quantity'] * df['priceUSDT']
        df.sort_values('USDT', 0, ascending=False, inplace=True)
        if hide_small_assets:
            df = df[df['USDT'] > 10]
        return df

    @staticmethod
    def _load_and_convert_ohlc(child, pairs, *args, **kwargs):
        supported = child.get_pairs()
        if isinstance(pairs, str):
            regex = re.compile(pairs)
            pairs = list(filter(regex.search, supported))
        print("%d pairs:" % len(pairs))

        pair_cache = {}
        cross_cache = {}
        dictionary = {}
        for pair in pairs:
            print("%s.. " % pair, end='')

            if pair in supported:
                if pair not in pair_cache:
                    pair_cache[pair] = child._get_ohlc(pair, *args, **kwargs)
                dictionary[pair] = pair_cache[pair]
            else:
                pair1, pair2, divide = Exchange._get_crosspairs(pair, supported)
                print("using %s cross %s.. " % (pair1, pair2), end='')

                if pair1 not in pair_cache:
                    pair_cache[pair1] = child._get_ohlc(pair1, *args, **kwargs)
                if pair2 not in cross_cache:
                    cross_cache[pair2] = child._get_ohlc(pair2, *args, **kwargs)
                dictionary[pair] = Exchange._convert_ohlc(pair_cache[pair1], cross_cache[pair2], divide)

            print("done")
        return dictionary

    @staticmethod
    def _load_and_convert_orderbooks(child, pairs, *args, **kwargs):
        ticker = child.get_ticker()
        if isinstance(pairs, str):
            regex = re.compile(pairs)
            pairs = list(filter(regex.search, ticker.keys()))
        print("%d pairs:" % len(pairs))

        pair_cache = {}
        dictionary = {}
        for pair in pairs:
            print("%s.. " % pair, end='')

            if pair in ticker:
                if pair not in pair_cache:
                    pair_cache[pair] = child._get_orderbook(pair, *args, **kwargs)
                dictionary[pair] = pair_cache[pair]
            else:
                pair1, pair2, divide = Exchange._get_crosspairs(pair, ticker)
                print("using %s cross %s.. " % (pair1, pair2), end='')

                if pair1 not in pair_cache:
                    pair_cache[pair1] = child._get_orderbook(pair1, *args, **kwargs)
                dictionary[pair] = Exchange._convert_orderbook(pair_cache[pair1], ticker[pair2], divide)

            print("done")
        return dictionary
