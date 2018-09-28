import tensorflow as tf


def metrics(yhat, y):
    loss = tf.losses.mean_squared_error(yhat, y)
    MAE_diff = tf.metrics.mean_absolute_error(y, yhat)
    MSE_diff = tf.metrics.mean_squared_error(y, yhat)
    y_total = tf.reduce_sum(y, axis=0)
    yhat_total = tf.reduce_sum(yhat, axis=0)
    MAE_total = tf.metrics.mean_absolute_error(y_total, yhat_total)
    MAPE_total = tf.metrics.mean(tf.div(tf.losses.absolute_difference(y_total, yhat_total), y_total))
    metricsdict = {
        "diff": {"MAE": MAE_diff,
                 "MSE": MSE_diff},
        "total": {"MAE": MAE_total,
                  "MAPE": MAPE_total}
    }
    return loss, metricsdict
