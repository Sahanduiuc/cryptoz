import pandas as pd


# https://stackoverflow.com/questions/7019283/automatically-type-cast-parameters-in-python
def boolify(s):
    if s == 'True':
        return True
    if s == 'False':
        return False
    raise ValueError("")


def autoconvert(s):
    if s is None:
        return None
    for fn in (boolify, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s


def stats(client):
    """Stats on symbols"""
    ticker = client.ticker(limit=10000)
    df = pd.DataFrame(ticker).set_index('symbol', drop=True)
    df = df.applymap(autoconvert)
    return df


def market_stats(client):
    """Stats on market"""
    return client.stats()
