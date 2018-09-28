import tensorflow as tf
from .predictor import seq2seq
from .opt import get_opt_step
Modekeys = tf.estimator.ModeKeys


def model_fn(x, y, mode, params):
    yhat = seq2seq(x, **params)

    if mode == Modekeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=yhat)

    loss = tf.losses.mean_squared_error(y, yhat)
    MAE_loss = tf.losses.mean_absolute_error(y, yhat)
    loss_summary = tf.summary.scalar("MSE_loss", loss)

    opt_step = get_opt_step(loss, **params)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=opt_step,
        eval_metric_ops={"MSE": loss, "MAE": MAE_loss})
