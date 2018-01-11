# based on https://github.com/ishai1/experiments/blob/master/RNN/gym.py
# tries mimics the kagglegym api, including scoring.

import os
import pandas as pd
from datetime import datetime as dt
import numpy as np
import math


def MAE(y_true, y_pred):
    return np.nanmean(np.abs(y_pred - y_true))


class Environment(object):
    def __init__(self, fpath, h, split=.5):
        """
        Parameters
        ----------
        self: type
            description
        fpath: string
            path to file containing training data
        h: int
            Number of time steps to prediction
        split: float
            Fraction of data to use for training.
        """
        assert h > 0
        df = pd.read_csv(fpath)
        df.time = pd.to_datetime(df.time)
        self.split = split
        self.h = int(h)

        df.sort_index('time', inplace=True)

        n = len(df)
        i = math.floor(n * self.split)
        self.n = n
        self.n_test = n - i     # number of test instances
        self.n_train = i        # number of train instances
        self.pretrain = df.loc[:i, :].copy()  # data provided at time 0
        self.test = df.loc[i:, :].reset_index(drop=True).copy()  # data to be provided iteratively
        del df

        # placeholder for predicted values in each time
        # y_hat[t] = predicted value at time t-h for response at t
        # initialized to true value so that unseen values will not affect MAE
        self.y_hat = self.test.price.copy()
        self.test['loss'] = np.NaN

    def reset(self):
        """
        Initialize the system and return the first observation as well as the training set
        """
        self.y_hat = self.test.price.copy()
        self.current_idx = 0  # at first only display MAE on new observations
        return self.train

    def step(self, target):
        """
        given current index t, yhat[t+h] = target
        """
        # assign target to panda predicted values along diagonal
        assert type(target) is float

        self.y_hat[self.h + self.current_idx] = target
        loss = np.abs(self.test.loc[self.current_idx, 'price'] - self.y_hat[self.current_idx])
        self.test.loc[self.current_idx, 'loss'] = loss
        current = self.test.loc[self.current_idx].drop('loss', axis=1).values
            # new observation
        self.current_idx += 1
        if self.current_idx > (self.n_test-(self.h[-1])):
            done = True
            # the total score is average MAE over the test set. Since each perdiction is for the
            # same number of time steps
            score = np.abs(self.test.price-self.test.y_hat).mean()
            info = {'final_score': score}
        else:
            done = False
            info = {}

        return current, loss, done, info

    def __str__(self):
        return "Environment()"

    def obs_shape(self):
        return self.train.shape[1]

    def target_shape(self):
        return self.test.shape[1]


def make(input_path):
    return Environment(input_path)
