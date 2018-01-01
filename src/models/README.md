- [Tasks](#org38e389c)
    - [Detrend](#org7b49567)
    - [Plot output](#org5ef5075)
    - [Remove periodicity where possible](#orgc91e3e6)
    - [Normalize using training data (?)](#orgbff6e0d)
  - [Analysis](#orgcd94edc)
    - [setup](#org2a8a8eb)
      - [Load](#orgf4b6581)
      - [model config](#org9b99d1d)
    - [explore](#orga3e1b1d)
    - [debug](#orgb4f66e1)
      - [run](#org121ca54)
      - [genertor](#orgb4c48db)
    - [run](#org4ed9da9)


<a id="org38e389c"></a>

# TODO Tasks


<a id="org7b49567"></a>

## Detrend

-   First difference + accumulate differences up to prediction time.


<a id="org5ef5075"></a>

## Plot output


<a id="orgc91e3e6"></a>

## Remove periodicity where possible


<a id="orgbff6e0d"></a>

## Normalize using training data (?)


<a id="orgcd94edc"></a>

# Analysis


<a id="org2a8a8eb"></a>

## setup


<a id="orgf4b6581"></a>

### Load

```ipython
import pandas as pd
import os 
from datetime import datetime as dt
df = pd.read_csv('../../data/data.csv')
df['time'] = pd.to_datetime(df.time)
df['timestamp'] = df.time.apply(dt.timestamp)
```

```ipython
from matplotlib import pyplot as plt
import numpy as np
data = df.groupby('timestamp').apply(lambda g: g[['price', 'size']].mean(axis=0))
assert np.diff(data.index).min() > 0  # data is sorted
```


<a id="org9b99d1d"></a>

### model config

```ipython
os.chdir('model1')
```

```ipython
from collections import namedtuple
flagdct = {'batch_size': 64,
	   'data_dir': '/tmp/dat/',
	   'hidden_dim': 200,
	   'l1reg_coeff': 1e-10,
	   'l2reg_coeff': 1e-9,
	   # 'l1reg_coeff': 1,
	   # 'l2reg_coeff': 1,
	   'latent_dim': 160,
	   'logdir': '/tmp/log/',
	   'n_epochs': 100000,
	   'n_iterations': 100000,
	   'n_samples_predict': 20,
	   'n_samples_train': 10,
	   'print_every': 1000, 
	   'huber_loss_delta': .1,
	   'use_update_ops': True}  # update_ops control dependency is necessary for batch norm
FLAGS = namedtuple('FLAGS',flagdct.keys())(**flagdct)
ff_params = dict(dim_hidden=20, dim_features=2, rnn_stack_height=3)  
```


<a id="orga3e1b1d"></a>

## explore

```ipython
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10,10))
for i, col in enumerate(data.columns.values):
   data.reset_index().plot.scatter(x='timestamp', y=col, ax=axes[i], s=.7)
```

```ipython
from pandas.plotting import scatter_matrix
scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
```


<a id="orgb4f66e1"></a>

## debug


<a id="org121ca54"></a>

### run

```ipython
import runner
catmodel = runner.Learner(ff_params, FLAGS)
```

```ipython
catmodel.fit(data.values, t_ix=data.index.values)
```

```ipython
from importlib import reload
reload(runner)
```


<a id="orgb4c48db"></a>

### genertor

```ipython
import utils
example_generator = utils.GrabSequence(X=data.values, t_ix=data.index.values, input_seq_len=100, time_gap_to_predict=10, stride=1)
```

```ipython
from importlib import reload
reload(utils)
```

```ipython
import tensorflow as tf
tf.reset_default_graph()
g = tf.Graph()
sess = tf.Session(graph=g)
with g.as_default():
    train_ds = tf.data.Dataset.from_generator(example_generator, (tf.float32, tf.float32),
					      (tf.TensorShape([None, 2]), tf.TensorShape([2])))
    train_ds.shuffle(buffer_size=100000)
    train_ds = train_ds.batch(100)
    iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
    training_init_op = iterator.make_initializer(train_ds)
    batch = iterator.get_next()
```

```ipython
with sess.as_default():
    sess.run(training_init_op)
    while (True):
	b = sess.run(batch)
```


<a id="org4ed9da9"></a>

## run

```ipython
from runner import Learner
catmodel = Learner(ff_params, FLAGS)
catmodel.fit(data.values, t_ix=data.index.values)
```