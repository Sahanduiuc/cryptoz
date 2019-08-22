from datetime import datetime
import pandas as pd

# Do not forget to install ipywidgets
from tqdm.autonotebook import tqdm

# For more details on Binance APIs:
# python-binance: https://python-binance.readthedocs.io/en/latest/index.html
# binance-official-api-docs: https://github.com/binance-exchange/binance-official-api-docs


class BinanceHelper():
    def __init__(self, client):
        self.client = client

    @staticmethod
    def ts_to_dt(ts):
        """Unix timestamp to datetime object"""

        return datetime.utcfromtimestamp(ts / 1000)

    @staticmethod
    def symbol_into_pair(symbol):
        """
        Parse Binance symbol to pair
        For example, BTCUSDT -> (BTC, USDT)
        """

        if isinstance(symbol, tuple):
            return symbol
        supported_quotes = ['USDT', 'BTC', 'ETH', 'BNB']
        for quote in supported_quotes:
            if symbol[-len(quote):] == quote:
                return symbol[:-len(quote)], quote
        return None

    @staticmethod
    def pair_into_symbol(pair):
        """
        Convert pair to Binance symbol
        For example, (BTC, USDT) -> BTCUSDT
        """

        if not isinstance(pair, tuple):
            return pair
        return ''.join(pair)

    def get_pairs(self):
        """Get the list of all pairs from Binance"""
        
        # Market Data Endpoints: https://python-binance.readthedocs.io/en/latest/market_data.html
        pairs = set(map(lambda d: self.symbol_into_pair(d['symbol']), self.client.get_all_tickers()))
        return list(filter(lambda s: s is not None, pairs))

    def get_ticker(self):
        """Get the latest price for all markets"""

        ticker = {}
        for d in self.client.get_all_tickers():
            pair = self.symbol_into_pair(d['symbol'])
            if pair is not None:
                ticker[pair] = float(d['price'])
        return ticker

    def get_active_balances(self, enrich=False):
        """Get the active, non-zero balances from the account information"""

        # Account Endpoints: https://python-binance.readthedocs.io/en/latest/account.html
        balances = self.client.get_account()['balances']
        df = pd.DataFrame(balances)
        df.set_index('asset', drop=True, inplace=True)
        df = df.astype(float)
        df = df[(df['free'] > 0) & (df['locked'] == 0)]
        df.drop('locked', 1, inplace=True)
        return df['free'].to_dict()

    @staticmethod
    def get_conversion_operation(pair, ticker):
        """
        Searches for pairs in the ticker required to build the requested pair.
        This way, for example, you can get the price of almost any possible trading pair.
        """

        base, quote = pair
        if base == quote:
            # BTC/BTC = 1.0
            return 1, 1, False
        elif (base, quote) in ticker:
            # NEO/BTC = 0.000933
            return 1, (base, quote), False
        elif (quote, base) in ticker:
            # USDT/BTC = 1 / (BTC/USDT) = 9.907287602614731e-05
            print(f"{base}/{quote} = 1 / ({quote}/{base})")
            return 1, (quote, base), True
        elif (base, 'BTC') in ticker and ('BTC', quote) in ticker:
            # GAS/USDT = (GAS/BTC) * (BTC/USDT) = 1.5846920599999998
            print(f"{base}/{quote} = ({base}/BTC) * (BTC/{quote})")
            return (base, 'BTC'), ('BTC', quote), False
        elif (base, 'BTC') in ticker and (quote, 'BTC') in ticker:
            # GAS/LSK = (GAS/BTC) / (LSK/BTC) = 1.3271344040574808
            print(f"{base}/{quote} = ({base}/BTC) / ({quote}/BTC)")
            return (base, 'BTC'), (quote, 'BTC'), True
        raise ValueError("Pair not supported")

    @staticmethod
    def calculate_pair_price(pair, ticker):
        """Calculate the price of the pair"""

        pair1, pair2, divide = BinanceHelper.get_conversion_operation(pair, ticker)
        if divide:
            operation = lambda x, y: x / y
        else:
            operation = lambda x, y: x * y
        if isinstance(pair1, tuple):
            factor1 = ticker[pair1]
        else:
            factor1 = pair1
        if isinstance(pair2, tuple):
            factor2 = ticker[pair2]
        else:
            factor2 = pair2
        return operation(factor1, factor2)

    def get_current_price(self, pair):
        """Get the current price of the pair"""

        ticker = self.get_ticker()
        return self.calculate_pair_price(pair, ticker)

    def get_active_balances_in(self, symbol):
        """Get the active balances in the currency specified (e.g. USDT)"""

        active_balances = self.get_active_balances()
        ticker = self.get_ticker()
        return {
            (from_symbol, symbol): quantity * self.calculate_pair_price((from_symbol, symbol), ticker) 
            for from_symbol, quantity in active_balances.items()
        }

    def _get_pair_ohlcv(self, pair, interval, **kwargs):
        """
        Get OHLCV data for one pair.
        Pair must be supported by Binance.
        """

        symbol = self.pair_into_symbol(pair)
        candles = self.client.get_klines(symbol=symbol, interval=interval, **kwargs)
        columns = ['date', 'O', 'H', 'L', 'C', '_', '_', 'V', '_', '_', '_', '_']

        df = pd.DataFrame(candles, columns=columns)
        df.set_index('date', drop=True, inplace=True)
        df.index = pd.to_datetime(df.index, unit='ms')
        df.fillna(method='ffill', inplace=True) # fill gaps forwards
        df.fillna(method='bfill', inplace=True) # fill gaps backwards
        df = df.astype(float)
        df = df[['O', 'H', 'L', 'C', 'V']]
        # https://www.thebalance.com/average-of-the-open-high-low-and-close-1031216
        df['M'] = (df['L'] + df['H'] + df['C']) / 3
        df = df.iloc[1:]  # first entry can be dirty
        return df

    def get_pair_ohlcv(self, pair, *args, **kwargs):
        """
        Wrapper around _get_pair_ohlcv with conversion logic (similar to price conversion)
        Can accept almost any pair of symbols as long as can be converted to BTC.
        For example, it can return OHLCV data for GAS in LSK units.
        """

        # Is the pair supported by Binance?
        ticker = self.get_ticker()
        if pair in ticker:
            return self._get_pair_ohlcv(pair, *args, **kwargs)

        # Do conversion otherwise
        pair1, pair2, divide = self.get_conversion_operation(pair, ticker)
        
        # Load the OHLCV data for both pairs
        ohlcv1_df = None
        ohlcv2_df = None
        if isinstance(pair1, tuple):
            # Pair can be built by BTC conversion
            ohlcv1_df = self._get_pair_ohlcv(pair1, *args, **kwargs)
        if isinstance(pair2, tuple):
            # Pair can be built by inversion
            ohlcv2_df = self._get_pair_ohlcv(pair2, *args, **kwargs)
        if ohlcv1_df is None and ohlcv2_df is None:
            raise ValueError("Pair not supported")

        # Convert OHLCV data
        if divide:
            operation = lambda x, y: x / y
        else:
            operation = lambda x, y: x * y
        if ohlcv1_df is None:
            # USDT/BTC = 1 / (BTC/USDT)
            ohlcv_df = ohlcv2_df.copy()
            ohlcv_df['O'] = operation(1, ohlcv2_df['O'])
            ohlcv_df['H'] = operation(1, ohlcv2_df['L']) # low becomes high
            ohlcv_df['L'] = operation(1, ohlcv2_df['H']) # high becomes low
            ohlcv_df['C'] = operation(1, ohlcv2_df['C'])
            ohlcv_df['V'] = operation(ohlcv2_df['V'], ohlcv2_df['M']) # normalize volume
            ohlcv_df['M'] = operation(1, ohlcv2_df['M']) # use average
        else:
            ohlcv_df = ohlcv1_df.copy()
            ohlcv_df['O'] = operation(ohlcv1_df['O'], ohlcv2_df['O'])
            ohlcv_df['H'] = operation(ohlcv1_df['H'], ohlcv2_df['M']) # use average
            ohlcv_df['L'] = operation(ohlcv1_df['L'], ohlcv2_df['M']) # use average
            ohlcv_df['C'] = operation(ohlcv1_df['C'], ohlcv2_df['C'])
            ohlcv_df['V'] = operation(ohlcv1_df['V'], ohlcv2_df['M']) # use average
            ohlcv_df['M'] = operation(ohlcv1_df['M'], ohlcv2_df['M']) # use average
        return ohlcv_df

    def get_multiple_pair_ohlcv(self, pairs, *args, **kwargs):
        """Fetch OHLCV data for multiple pairs"""

        ohlcvs = {}
        pbar = tqdm(pairs)
        for pair in pbar:
            ohlcvs[pair] = self.get_pair_ohlcv(pair, *args, **kwargs)
            pbar.set_description("Processing %s" % self.pair_into_symbol(pair))
        return ohlcvs

    def _get_pair_depth(self, pair, **kwargs):
        """
        Get order book and construct cumulative depth (as can be seen in Binance under the Depth tab)
        Returns a series of cumulative bids and asks (negative sign) indexed by the rate.
        Both amounts and rates are in quote currency, for multiple pairs to be comparable.
        Meant to visualize supply and demand at different prices.
        Pair must be supported by Binance.
        """

        # Get order book
        symbol = self.pair_into_symbol(pair)
        order_book = self.client.get_order_book(symbol=symbol, **kwargs)

        # Cumulative bids
        rates, amounts = zip(*order_book['bids'])
        cum_bids = pd.Series(amounts, index=rates, dtype=float)
        cum_bids.index = cum_bids.index.astype(float)
        cum_bids *= cum_bids.index
        cum_bids = cum_bids.sort_index(ascending=False).cumsum().sort_index()

        # Cumulative asks
        rates, amounts = zip(*order_book['asks'])
        cum_asks = pd.Series(amounts, index=rates, dtype=float)
        cum_asks.index = cum_asks.index.astype(float)
        cum_asks *= cum_asks.index
        cum_asks = -cum_asks.sort_index().cumsum()

        # Append both series into a continuous one with asks being negative values
        return cum_bids.append(cum_asks).sort_index()

    def get_pair_depth(self, pair, **kwargs):
        """
        Wrapper around _get_pair_depth with conversion logic (similar to price conversion)
        Can accept almost any pair of symbols as long as can be converted to BTC.
        For example, it can return depth for GAS in LSK units.
        """

        # Is the pair supported by Binance?
        ticker = self.get_ticker()
        if pair in ticker:
            return self._get_pair_depth(pair, **kwargs)

        # Do conversion otherwise
        pair1, pair2, divide = self.get_conversion_operation(pair, ticker)
        
        # Load the order book for one pair, and use the second pair for getting the conversion price
        if isinstance(pair1, tuple):
            # Pair can be built by BTC conversion
            depth_sr = self._get_pair_depth(pair1, **kwargs)
        elif isinstance(pair2, tuple):
            # Pair can be built by inversion
            depth_sr = self._get_pair_depth(pair2, **kwargs)
        else:
            raise ValueError("Pair not supported")

        # Convert rates and amounts according to the conversion operation
        if divide:
            operation = lambda x, y: x / y
        else:
            operation = lambda x, y: x * y
        depth_sr = operation(depth_sr, ticker[pair2])
        depth_sr.index = operation(depth_sr.index, ticker[pair2])
        return depth_sr
