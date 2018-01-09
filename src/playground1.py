# based on https://github.com/ishai1/experiments/blob/master/RNN/gym.py
# tries mimics the kagglegym api, including scoring.

import numpy as np
import math


class Environment(object):
    def __init__(self, data, h, split=.5):
        """
        Parameters
        ----------
        self: type
            description
        data: pd.DataFrame
            Data frame containing training and test data.
        h: int
            Number of time steps to prediction
        split: float
            Fraction of data to use for training.
        """
        assert h > 0
        self.split = split
        self.h = int(h)
        self.price_index = 1  # Index in data for response variable

        n = len(data)
        i = math.floor(n * self.split)
        self.n = n
        self.n_test = n - i     # number of test instances
        self.n_train = i        # number of train instances
        self.pretrain = data[:i, :].copy()  # data provided at time 0
        self.test = data[i:, :].copy()  # data to be provided iteratively
        del data

        # placeholder for predicted values in each time
        # y_hat[t] = predicted value at time t-h for response at t
        # initialized to true value so that unseen values will not affect MAE
        self.y_hat = self.test[:, self.price_index].copy()
        self.test_loss = np.array([np.nan] * self.y_hat.shape[0])  # loss for each time step

    def reset(self):
        """
        Initialize the system and return the first observation as well as the training set
        """
        self.y_hat = self.y_hat = self.test[:, self.price_index].copy()
        self.current_idx = 0  # at first only display MAE on new observations
        return self.pretrain

    def step(self, target):
        """
        given current index t, y_hat[t+h] = target
        """
        # assign target to panda predicted values along diagonal
        target = float(target)

        self.y_hat[self.h + self.current_idx] = target
        loss = np.abs(self.test[self.current_idx, self.price_index] - self.y_hat[self.current_idx])
        self.test_loss[self.current_idx] = loss
        current = self.test[self.current_idx, :]  # new observation
        self.current_idx += 1
        if self.current_idx > (self.n_test - self.h):
            done = True
            # the total score is average MAE over the test set. Since each perdiction is for the
            # same number of time steps
            score = np.abs(self.test.price - self.test.y_hat)
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
