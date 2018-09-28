import tensorflow as tf


def get_opt_step(loss, lr_start=1e-4, lr_decay=.999995, clip_norm=50., **kwargs):
    global_step = tf.train.get_global_step()
    lr_decay = tf.Variable(lr_decay, trainable=False)
    decay_steps = tf.Variable(1, trainable=False)
    learning_rate = tf.train.exponential_decay(lr_start, global_step, decay_steps,
                                               lr_decay, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_norm(grad, clip_norm), var) for (grad, var) in gvs]
    return optimizer.apply_gradients(capped_gvs, global_step=global_step)
