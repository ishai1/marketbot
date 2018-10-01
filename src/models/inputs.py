import tensorflow as tf
from .utils import normalize
import numpy as np
import os

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
    targets = tf.expand_dims(_pct_change(data[:, 1], horizon), axis=1)
    targets = normalize(targets, name='targets', mode=ModeKeys.TRAIN)
    data = data[:-horizon]  # Remove last horizon observations which lack a corresponding response value
    data = normalize(data, name='observations', mode=ModeKeys.TRAIN)
    data = tf.concat([data, targets], axis=1)

    data = _rolling_windows(data, window)

    dataset = tf.data.Dataset.from_tensor_slices((data[:, :, :-1], data[:, :, -1]))

    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset


def _eval_input(data, horizon):
    targets = _pct_change(data[:, 1], horizon)
    targets = tf.expand_dims(targets, axis=1)
    targets = tf.squeeze(normalize(targets, name='targets', mode=ModeKeys.EVAL))

    data = data[:-horizon]
    data = normalize(data, name='observations', mode=ModeKeys.EVAL)

    return tf.data.Dataset.from_tensor_slices((tf.expand_dims(data, axis=0),
                                               tf.expand_dims(targets, axis=0))).batch(1)


def _predict_input(data):
    data = normalize(data, name='observations', mode=ModeKeys.PREDICT)
    return tf.data.Dataset.from_tensor_slices(tf.expand_dims(data, axis=0)).batch(1)


def _rolling_windows(data, window):
    return tf.contrib.signal.frame(data, window, 1, axis=0)


def _log_difference(feature_col, horizon):
    log_feature = tf.log(feature_col)
    return log_feature[horizon:] - feature_col[: -horizon]


def _pct_change(feature_col, horizon):
    return tf.div(feature_col[horizon:], feature_col[: -horizon]) - 1


def _read_csv_to_tensor(path):
    data = np.genfromtxt(path, delimiter=',')[2:, 1:]
    return tf.convert_to_tensor(data, dtype=tf.float32)

