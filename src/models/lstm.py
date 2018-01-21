"""
model definition for LSTM
"""
import tensorflow as tf
import numpy as np

ModeKeys = tf.estimator.ModeKeys

DEFAULT_TRAIN_PARAMS = {
    'window': 100,
    'num_epochs': 1000,
    'batch_size': 10
}

DEFAULT_MODEL_PARAMS = {
    'lstm_activation': tf.nn.tanh,
    'dense_activation': None,
    'loss': tf.losses.mean_squared_error,
    'learning_rate': 0.001,
    'layer_sizes': [10, 10, 10],
    'optimizer': tf.train.AdamOptimizer,
    'scale_l1': 1e-3,
    'scale_l2': 1e-3
}


MODEL_OUTPUT_DIR = 'trained_models'


def _rnn_model_fn(features, labels, mode, params):
    # LSTM cell builder function
    prices = features[:, :, 1]

    lstm_layer_fn = lambda size: tf.nn.rnn_cell.BasicLSTMCell(
        size, activation=params['lstm_activation'])


    # Stack LSTM cells
    lstm_stack = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_layer_fn(i) for i in params['layer_sizes']])

    # Feed forward through stacked LSTM
    lstm_out, _ = tf.nn.dynamic_rnn(cell=lstm_stack, inputs=features, dtype=tf.float32)

    # Feed lstm output through dense layer to get prediction
    dense_out = tf.layers.dense(
        inputs=lstm_out,
        units=1,
        activation=params['dense_activation'],
        name='dense')

    predictions = tf.squeeze(dense_out, axis=-1)

    if mode == ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"predictions": predictions})

    # Compute loss
    loss = tf.reduce_mean(params['loss'](predictions, labels))
    scale_l1, scale_l2 = params['scale_l1'], params['scale_l2']
    regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1, scale_l2)
    trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = [v for v in trainables if 'bias' not in v.name]
    reg_loss = tf.contrib.layers.apply_regularization(regularizer, kernels)

    optimizer = params['optimizer'](learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(
        loss=reg_loss, global_step=tf.train.get_global_step())

    # eval matric ops
    predictions_and_prices = tf.stack([predictions, prices], axis=2)
    pnl_fn = lambda t: tf.cond(
        t[0] > 0,
        true_fn=lambda: (-1., 1 / t[1]),
        false_fn=lambda: (1., -1 / t[1]))

    account_movement = tf.map_fn(
        lambda w: tf.map_fn(pnl_fn, w, dtype=(tf.float32, tf.float32)),
        predictions_and_prices,
        dtype=(tf.float32, tf.float32))

    weights = tf.stack([tf.ones([tf.shape(prices)[0]]), prices[:, -1]], axis=1)

    pnl = tf.reduce_sum(
        weights * tf.transpose(account_movement),
        axis=1)

    eval_metric_ops = {
        'pnl': tf.metrics.mean(pnl)
    }


    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def _input_fn_wrapper(path, mode, horizon, train_params=None):
    def _input_fn():
        data = _read_csv_to_tensor(path)
        if mode == ModeKeys.TRAIN:
            data = _train_input(data, horizon, **train_params)
        elif mode == ModeKeys.EVAL:
            data = _eval_input(data, horizon)
        else:
            data = _predict_input(data)
        return data
    return _input_fn


def _train_input(data, horizon, num_epochs, batch_size, window):
    targets = _pct_change(data[:, 1], horizon)
    data = tf.concat([data[:-horizon], tf.reshape(targets, [-1, 1])], axis=1)
    data = _rolling_windows(data, window)

    dataset = tf.data.Dataset.from_tensor_slices(
        (data[:, :, :-1], data[:, :, -1]))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset


def _eval_input(data, horizon):
    targets = _pct_change(data[:, 1], horizon)
    data = data[:-horizon]
    return tf.expand_dims(data, axis=0), tf.expand_dims(targets, axis=0)

def _predict_input(data):
    return tf.expand_dims(data, axis=0)

def _rolling_windows(data, window):
    return tf.contrib.signal.frame(data, window, 1, axis=0)

def _pct_change(feature_col, horizon):
    return tf.div(feature_col[horizon:], feature_col[: -horizon]) - 1

def _read_csv_to_tensor(path):
    data = np.genfromtxt(path, delimiter=',')[2:, 1:]
    return tf.convert_to_tensor(data, dtype=tf.float32)


def estimator(params=None, model_dir=None):
    if not params:
        params = DEFAULT_MODEL_PARAMS
    if not model_dir:
        model_dir = MODEL_OUTPUT_DIR
    rnn = tf.estimator.Estimator(model_fn=_rnn_model_fn,
                                 params=params,
                                 model_dir=model_dir)
    return rnn

def train(rnn, path):
    train_input_fn = _input_fn_wrapper(path,
                                       ModeKeys.TRAIN,
                                       10,
                                       DEFAULT_TRAIN_PARAMS)
    rnn.train(input_fn=train_input_fn)

def evaluate(rnn, path):
    eval_input_fn = _input_fn_wrapper(path, ModeKeys.EVAL, 10)
    rnn.evaluate(input_fn=eval_input_fn, steps=1)

def main():
    rnn = estimator()
    train(rnn, 'data/clean/data3.csv')
