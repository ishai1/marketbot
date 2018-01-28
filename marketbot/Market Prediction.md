

```python
import pandas as pd
import matplotlib.pyplot as plt

```

Our data is a series of BTC-USD trades made on GDAX. The features are described below:

 feature | description 
 ------- | ------------
 price   | the exchange rate for the trade - i.e., USD per Bitcoin
 size    | the size of the trade
 side    | whether the order was executed by a 'buyer' or 'seller'
 time    | the time at which GDAX notified us of the trade 


The first thing we notice is that at a given instant in time, there are multiple simultaneous orders which come through. Our first task will be to clean up the data a bit and solove this issue.


```python
df = pd.read_csv('data/data2.csv', index_col='sequence')
df['time'] = pd.to_datetime(df['time']).astype(int)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>size</th>
      <th>side</th>
      <th>time</th>
    </tr>
    <tr>
      <th>sequence</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4724622867</th>
      <td>13021.43</td>
      <td>0.001190</td>
      <td>sell</td>
      <td>1514708107700000000</td>
    </tr>
    <tr>
      <th>4724622869</th>
      <td>13021.43</td>
      <td>0.000346</td>
      <td>sell</td>
      <td>1514708107700000000</td>
    </tr>
    <tr>
      <th>4724622871</th>
      <td>13021.52</td>
      <td>0.001190</td>
      <td>sell</td>
      <td>1514708107700000000</td>
    </tr>
    <tr>
      <th>4724622873</th>
      <td>13021.53</td>
      <td>0.001958</td>
      <td>sell</td>
      <td>1514708107700000000</td>
    </tr>
    <tr>
      <th>4724622875</th>
      <td>13021.54</td>
      <td>0.002645</td>
      <td>sell</td>
      <td>1514708107700000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped = df.groupby('time')

def normalize_price(g):
    vol = g['size'].sum()
    price = (g['size'] * g.price).sum() / vol
    return pd.Series([vol, price], ['size', 'price'])

agg_df = grouped.apply(normalize_price).reset_index()
```


```python
agg_df.plot(x='time', y='price')
df.plot(x='time', y='price')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x114c780b8>




![png](Market%20Prediction_files/Market%20Prediction_4_1.png)



![png](Market%20Prediction_files/Market%20Prediction_4_2.png)

