import tensorflow as tf
import pandas as pd

def window_gen(path, horizon, window):
    df = pd.read_csv(path)
    outcomes = df.price.pct_change(periods=horizon).dropna().as_matrix()
    df = df[['volume', 'change']].as_matrix()

    for i in range(1, len(df) - horizon - window):
        yield (df[i: i + window], outcomes[i + window])
