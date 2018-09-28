import tensorflow as tf
import numpy as np


def input_wrapper_function(data, len_in, len_out, nbatch, nepochs):
    end_ix = data.shape[1] - (len_in + len_out)
    start_indices = np.arange(end_ix)

    tdata = tf.constant(data, dtype=tf.float32)
    train_ds = tf.data.Dataset.from_tensor_slices(start_indices)
    train_ds = train_ds.shuffle(buffer_size=40000)

    def get_input_output(ix):
        range_in = tf.range(ix, ix + len_in)
        range_out = tf.range(ix + len_in, ix + len_in + len_out)
        return tf.gather(tdata, range_in, axis=0), tf.gather(tdata, range_out, axis=0)

    train_ds = train_ds.map(get_input_output, num_parallel_calls=8).prefetch(5000)
    if nbatch is None:
        nbatch = start_indices.shape[0]
    train_ds = train_ds.batch(nbatch).repeat(nepochs)

    def _input_fn():
        return train_ds.make_one_shot_iterator().get_next()

    return _input_fn
