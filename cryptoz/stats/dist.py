import pandas as pd
import numpy as np

def _percentiles(sr, min, max, step):
    index = range(min, max + 1, step)
    return pd.Series({x: np.nanpercentile(sr, x) for x in index})


def percentiles(df, *args):
    return df.apply(lambda sr: _percentiles(sr, *args))
