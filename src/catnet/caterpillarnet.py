import tensorflow as tf
Dense = tf.layers.Dense


class FeedForward(object):
    """
                                         +-
                                         | \-pred_price
                                       +-+ /-pred_volume
                                       | +-
                                       |
      +----+  +----+  +----+        +--+-+
      |    |  |    |  |    |        |    |
      |    +--+    +--+    +-oo  oo-+    |
      +-+--+  +--+-+  +--+-+        +--+-+
        |        |       |             |
        |        |       |             |
        x_1      x_2     x_3           x_n

    It's a caterpillar!
    """
    def __init__(self, dim_hidden, dim_features, rnn_stack_height=3):
        self.dim_fetaures = dim_features
        self.dim_hidden = dim_hidden
        with tf.variable_scope('rnn'):
            def lstm_cell(nunits):
                return tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden, activation=tf.nn.elu)
            self.cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(dim_hidden) for
                                                        i in range(rnn_stack_height)])
            self.denseout = Dense(dim_features, activation=tf.relu)

    def __call__(self, _x):
        """
        Parameters
        ----------
        _x: tf.Tensor
            batched input passed to model. _x.shape is (B x T x D).

        Returns
        -------
        tf.Tensor
        """
        with self.g.as_default():
            _x = tf.transpose(_x, [1, 0, 2])  # to time major
            h_out, _ = tf.nn.dynamic_rnn(cell=self.cell_fw, inputs=_x,
                                         initial_state=None,
                                         time_major=True, dtype=tf.float32)
            return self.denseout(h_out[-1, :, :])  # predict based on final hidden output
