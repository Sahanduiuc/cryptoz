import re


class Exchange(object):
    def __init__(self, client):
        self.client = client

    def get_pairs(self):
        pass

    def get_price(self, pair):
        pass

    def get_ohlc(self, pairs, *args, **kwargs):
        pass

    def get_orderbooks(self, pairs, *args, **kwargs):
        pass

    @staticmethod
    def _convert_ohlc(ohlc, cross_ohlc, divide=True):
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

    @staticmethod
    def _convert_orderbook(orderbook, cross_price, divide=True):
        if divide:
            cross_price = 1 / cross_price
        orderbook = orderbook.copy()
        orderbook *= cross_price
        orderbook.index *= cross_price
        return orderbook

    def _load_pairs(self, pairs, convert_func, load_func, cross_func=None):
        """
        Convert and load pairs

        :param supported: pairs supported by exchange
        :param convert_func: how to convert pair
        :param load_func: how to load pair
        :param cross_func: which data to fetch for cross pair (price/historical price/etc)
        :return:
        """

        supported = self.get_pairs()

        if isinstance(pairs, str):
            regex = re.compile(pairs)
            pairs = list(filter(regex.search, supported))

        print("%d pairs:" % len(pairs))

        cross_cache = {}
        dictionary = {}
        for pair in pairs:
            print("%s.. " % pair, end='')
            base, quote = pair.split('/')
            if pair in supported:
                dictionary[pair] = load_func(pair)
            else:
                if quote + '/BTC' in supported:
                    cross_pair = quote + '/BTC'
                    divide = True
                elif 'BTC/' + quote in supported:
                    cross_pair = 'BTC/' + quote
                    divide = False
                else:
                    raise Exception("Cross pair not found.")

                if cross_pair not in cross_cache:
                    if cross_func is None:
                        cross_func = load_func
                    cross_cache[cross_pair] = cross_func(cross_pair)
                print("cross %s.. " % cross_pair, end='')
                dictionary[pair] = convert_func(load_func(base + '/BTC'), cross_cache[cross_pair], divide=divide)

            print("done")
        return dictionary
