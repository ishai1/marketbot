import pandas as pd

def get_data(path, interval_length=1, stride_length=10):
    df = pd.read_csv(path, index_col='sequence')
    df['time'] = pd.to_datetime(df['time'])
    agg_df = discretize(df, interval_length)

    agg_df.volume = agg_df.volume.fillna(0.0)
    agg_df.price = agg_df.price.fillna(method='ffill')
    agg_df['change'] = agg_df.price.pct_change()
    agg_df['outcome']= agg_df.price.pct_change(periods=(-1 * stride_length))

    return agg_df


def discretize(df, interval_length):
    """ interval_length = number of seconds to aggregate in """
    if interval_length == 0:
        grouped = df.groupby('time')
    else:
        grouped = df.resample('{}S'.format(interval_length), on='time')
    return grouped.apply(mean_price).reset_index()

def mean_price(g):
    """ computes size-weighted price of trades """
    vol = g['size'].sum()
    price = (g['size'] * g['price']).sum() / vol
    return pd.Series([vol, price], ['volume', 'price'])
