import tensorflow as tf
nn = tf.nn
dense = tf.layers.dense
ModeKeys = tf.estimator.ModeKeys


def seq2seq(seqin, xstart, Nh=10, Nc=12, D=None, **kwargs):
    Nfwd = Nh // 2
    Nbwd = Nh // 2
    forward = nn.rnn_cell.LSTMCell(Nfwd)
    backward = nn.rnn_cell.LSTMCell(Nbwd)

    seqin_s0 = tf.layers.dense(xstart, Nfwd * 2)
    seqin_s0 = nn.rnn_cell.LSTMStateTuple(c=seqin_s0[:, :Nfwd], h=seqin_s0[:, Nfwd:])

    len_out = kwargs.get('len_out')
    if not len_out:
        raise Exception("invalid value for len_out")
    outputs, output_states = nn.bidirectional_dynamic_rnn(forward, backward, seqin, time_major=True,
                                                          initial_state_fw=seqin_s0, dtype=tf.float32)
    statfw, statbw = output_states
    enc = tf.concat(outputs, axis = -1)
    statfw = tf.concat(statfw, 1)  # concatenate
    state_init = tf.layers.dense(statfw, 2 * Nh)

    # Transfer state to output rnn and split
    ct = state_init[:, :Nh]
    ht = state_init[:, Nh:2 * Nh]

    recur = nn.rnn_cell.LSTMCell(Nh)

    def align(t):
        T = tf.tile(tf.expand_dims(t, 0), [tf.shape(enc)[0], 1, 1])
        stilde = tf.layers.dense(tf.concat([T, enc], -1), Nc, activation = tf.tanh)
        return nn.softmax(tf.squeeze(tf.layers.dense(stilde, 1, use_bias=False), axis=2), axis=0)

    def body(i, s, y_tm1, ta):
        """
        predict based on previous hidden state and prediction
        i: current timestep
        s: last step's state. shape = ((B x n_hidden), (B x n_hidden))
        ta: dynamic tensorarray to store outputs (can be static, atm)
        """
        alpha = align(s.h)
        c = tf.reduce_sum(tf.multiply(enc, tf.expand_dims(alpha, 2)), axis=0)
        ytilde = tf.concat([y_tm1, c], axis=-1)  # concatenate context and previous
        y_t, s = recur(tf.layers.dense(ytilde, Nh), s)
        ta = ta.write(i, y_t)
        return i + 1, s, y_t, ta

    cond = lambda i, s, y, ta: i < len_out
    i0 = tf.constant(0, dtype=tf.int32)
    y0 = enc[-1, :, :]  # tf.zeros_like(enc[0,:,:])
    s0 = nn.rnn_cell.LSTMStateTuple(c=ct, h=ht)
    ta0 = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True,
                         element_shape=tf.TensorShape([None, Nh]))  # array of tensors

    i, s, y, ta = tf.while_loop(cond, body, [i0, s0, y0, ta0])
    return tf.layers.dense(ta.stack(), D)
