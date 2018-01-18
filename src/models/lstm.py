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
    'lstm_activation': tf.nn.relu,
    'dense_activation': tf.nn.relu,
    'loss': tf.losses.mean_squared_error,
    'learning_rate': 0.001,
    'layer_sizes': [20, 40, 80],
    'optimizer': tf.train.AdamOptimizer
}


MODEL_OUTPUT_DIR = 'trained_models'


def _rnn_model_fn(features, labels, mode, params):
    # LSTM cell builder function
    lstm_layer_fn = lambda size: tf.nn.rnn_cell.BasicLSTMCell(
        size,
        activation=params['lstm_activation'])

    # Stack LSTM cells
    lstm_stack = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_layer_fn(i) for i in params['layer_sizes']])

    # Feed forward through stacked LSTM
    lstm_out, _ = tf.nn.dynamic_rnn(cell=lstm_stack,
                                    inputs=features,
                                    dtype=tf.float32)

    # Feed lstm output through dense layer to get prediction
    dense_out = tf.layers.dense(
        inputs=lstm_out,
        units=1,
        activation=params['dense_activation'],
        name='dense')

    predictions = tf.squeeze(dense_out)

    if mode == ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"predictions": predictions})

    loss = params['loss'](predictions, labels)
    optimizer = params['optimizer'](learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op)


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
    data = data[:horizon]
    return tf.reshape(data[:horizon], [1, -1, 3]), tf.reshape(targets, [1, -1])

def _predict_input(data):
    return tf.reshape(data, [1, -1, 3])

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
    rnn.evaluate(input_fn=eval_input_fn)

def main():
    rnn = estimator()
    train(rnn, 'data/clean/data3.csv')
