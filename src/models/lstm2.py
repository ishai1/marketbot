"""
model definition for LSTM
"""
import tensorflow as tf

from .inputs import _input_fn_wrapper

ModeKeys = tf.estimator.ModeKeys

DEFAULT_TRAIN_PARAMS = {
    'window': 100,
    'num_epochs': 500,
    'batch_size': 10
}

DEFAULT_MODEL_PARAMS = {
    'lstm_activation': tf.nn.tanh,
    'dense_activation': None,
    'loss': tf.losses.mean_squared_error,
    'learning_rate': 0.01,
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
        name='dense'  # ,
        # kernel_initializer=tf.keras.initializers.glorot_uniform()
    )

    predictions = tf.squeeze(dense_out, axis=-1)

    if mode == ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"predictions": predictions})

    loss = tf.reduce_mean(params['loss'](predictions, labels))

    scale_l1, scale_l2 = DEFAULT_MODEL_PARAMS['scale_l1'], DEFAULT_MODEL_PARAMS['scale_l2']
    regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1, scale_l2)
    trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = [v for v in trainables if 'bias' not in v.name]
    reg_loss = tf.contrib.layers.apply_regularization(regularizer, kernels)

    optimizer = params['optimizer'](learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss=loss + reg_loss, global_step=tf.train.get_global_step())

    with tf.variable_scope('normalize/targets', reuse=True):
        std_coeffs = tf.get_variable('std')
    x = tf.squeeze(predictions) * std_coeffs
    y = tf.squeeze(prices) * std_coeffs
    pnl = tf.where(x > 0, -1 + 1.0 / y, 1 - 1.0 / y)

    eval_metric_ops = {
        'pnl': tf.metrics.mean(pnl)
    }
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def estimator(params=None, model_dir=None):
    if not params:
        params = DEFAULT_MODEL_PARAMS
    if not model_dir:
        model_dir = MODEL_OUTPUT_DIR
    rnn = tf.estimator.Estimator(model_fn=_rnn_model_fn,
                                 params=params,
                                 model_dir=model_dir)
    return rnn


def train(rnn, path, steps=None):
    train_input_fn = _input_fn_wrapper(path,
                                       ModeKeys.TRAIN,
                                       10,
                                       DEFAULT_TRAIN_PARAMS)
    rnn.train(input_fn=train_input_fn, steps=steps)


def evaluate(rnn, path):
    eval_input_fn = _input_fn_wrapper(path, ModeKeys.EVAL, 10)
    rnn.evaluate(input_fn=eval_input_fn, steps=1)


def main():
    rnn = estimator()
    for i in range(100):
        train(rnn, 'data/clean/data3.csv', steps=200)
        evaluate(rnn, 'data/clean/data.csv')
