import tensorflow as tf
import os

ModeKeys = tf.estimator.ModeKeys


def denormalize(X, name):
    """
    Reverse normalization across axis 0 for variable X.
    Normalization params are collected from scope = name.
    """
    with tf.variable_scope(os.path.join('normalize', name), reuse=True):
        std_coeffs = tf.get_variable('std')
        mu_coeffs = tf.get_variable('mean')
    return X * std_coeffs + mu_coeffs


def normalize(X, name, mode):
    """
    Apply normalization across axis 0 for variable X.
    Normalization params are collected from scope = name.
    """
    with tf.variable_scope(os.path.join('normalize', name)):
        if mode == ModeKeys.TRAIN:
            mu, std = calc_mu_std(X)
        else:
            shape = (X.shape[1],)
            mu = tf.get_variable('mean', shape=shape, trainable=False)
            std = tf.get_variable('std', shape=shape, trainable=False)
    return tf.div(X - mu, std + 1e-8)


def calc_mu_std(X):
    """
    Calculate mean and standard devation and store for later use.
    Use assign instead of initalize in-case the mu / std already exist.
    """
    mu0, var = tf.nn.moments(X, axes=(0))
    shape = (mu0.shape[0], )
    mu = tf.get_variable('mean', shape=shape, trainable=False)
    mu.assign(mu0)
    std = tf.get_variable('std', shape=shape, trainable=False)
    std.assign(tf.sqrt(var))
    return mu, std
