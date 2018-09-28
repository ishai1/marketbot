import tensorflow as tf
from .predictor import seq2seq
from .opt import get_opt_step
from .loss import metrics
ModeKeys = tf.estimator.ModeKeys


def model_fn(features, labels, mode, params):

    device = tf.device('/cpu:0') if mode != ModeKeys.TRAIN else tf.device("/device:GPU:0")
    with device:
        x = features["xdiff"]
        xstart = features["xin"]
        y = labels
        x = tf.transpose(x, [1, 0, 2])  # x is batch major but seq2seq assumes time major
        yhat = seq2seq(x, xstart, **params)
        yhat = tf.transpose(yhat, [1, 0, 2])  # and back to batch major

        if mode == ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=yhat)

        loss, metricsdict = metrics(yhat, y)
        loss_summary = tf.summary.scalar("MSE_loss", loss)
        opt_step = get_opt_step(loss, **params)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=opt_step,
        eval_metric_ops=metricsdict)
