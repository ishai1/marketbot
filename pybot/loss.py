import tensorflow as tf


def metrics(yhat, y):
    loss = tf.losses.mean_squared_error(yhat, y)
    MAE_diff = tf.metrics.mean_absolute_error(y, yhat)
    MSE_diff = tf.metrics.mean_squared_error(y, yhat)
    y_total = tf.reduce_sum(y, axis=1)  # y and yhat are batch major
    yhat_total = tf.reduce_sum(yhat, axis=1)
    MAE_total = tf.metrics.mean_absolute_error(y_total, yhat_total)
    MAPE_total = tf.metrics.mean(tf.div(tf.losses.absolute_difference(y_total, yhat_total), y_total))
    metricsdict = {
        "MAE_diff": MAE_diff,
        "MSE_diff": MSE_diff,
        "MAE_total": MAE_total,
        "MAPE_total": MAPE_total}

    return loss, metricsdict
