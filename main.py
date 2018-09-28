import tensorflow as tf
import pandas as pd
from pybot.model import model_fn
from pybot.input import input_wrapper_function
tf.logging.set_verbosity(tf.logging.DEBUG)


def main(data, model_dir=None, **params):
    input_fn = input_wrapper_function(df.values, **params)
    config = tf.estimator.RunConfig(save_summary_steps=1000,
                                    tf_random_seed=1234,
                                    save_checkpoints_secs=600)
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params=params,
                                       model_dir=model_dir,
                                       config=config)
    estimator.train(input_fn)


if __name__ == "__main__":
    df = pd.read_feather("postproc3.feather").drop("datevalue", axis=1)
    main(df.values, model_dir="tmpZc797s", len_in=100, len_out=20, nbatch=32, nepochs=10, D=df.shape[1])
