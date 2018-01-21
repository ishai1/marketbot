import tensorflow as tf
import numpy as np

ModeKeys = tf.estimator.ModeKeys


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
