# based on https://github.com/ishai1/experiments/blob/master/RNN/gym.py
# This file mimics the kagglegym api, including scoring.
#

import pandas as pd
import numpy as np
import math

def MAE(y_true, y_pred):
    return np.nanmean(np.abs(y_pred-y_true))

class Environment(object):
    def __init__(self, fpath, h, split = .5, use_diff = False):
        """
        h -- time horizon to be predicted (ex. np.arange(9,49)). Should be numpy array
        split -- what fraction to use for trainnig
        """
        assert isinstance(h,np.ndarray)
        data = pd.read_csv(fpath)
        self.split = split
        self.h = h

        if 'Formatted Date' in data.columns:
            dt = pd.to_datetime(data['Formatted Date'])  # adjust dates to unix
            dt = (dt.astype(np.int64)/10**9)
            data['timestamp'] = (dt/3600).astype(np.int32)  # hourly
        data.drop(data.columns[data.dtypes==object], axis=1, inplace=True)

        changes = {'Temperature (C)':'y'}
        data.columns = [changes[c] if c in changes.keys() else c for c in data.columns]

        # Get a list of unique timestamps
        # use the first half for training and
        # the second half for the test set
        if ('timestamp' in data.columns) and (not data.timestamp.is_unique):
            print("Found duplicate time entries, dropping latest.")
            data.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
            assert (np.sum(np.square(
                data[data.duplicated(['timestamp'], keep='first')].values-
                data[data.duplicated(['timestamp'], keep='last')].values)) == 0)
            data.sort_values(by='timestamp', inplace=True)
            if not data.index.is_monotonic:
                print("non monotonic index, resetting")
                data.reset_index(inplace=True)
            assert data.index.is_monotonic
        if use_diff:
            data = data.diff()[1:]  # working with first differences is a little nicer
            assert all(data.timestamp>0)
            data.reset_index(drop=True,inplace=True)
            if not all(data.timestamp>0):
                print("non unary timestamp increase at ", data.timestamp[data.timestamp!=1])
            data.drop(labels='timestamp', axis=1, inplace=True)
        n = len(data)
        i = math.floor(n*self.split)
        self.n = n
        self.n_test = n-i
        self.n_train = i
        self.train = data[:i].copy()
        self.test = data[i:].reset_index(drop=True).copy()
        del data

        # probably a better way to do this... convert back to DF to help with diagonal slicing later
        self.y_hat = np.vstack(self.test.apply(lambda r: [r['y']]*len(self.h),axis=1))\
            # placeholder for predicted values in each time\
            # y_hat[t,k] = predicted value at time t-h[k] for response at t\
            # initialized to true value so that unseen values will not affect MAE
        self.test['loss'] = np.NaN

    def reset(self):
        """
        Initialize the system and return the first observation as well as the training set
        """
        self.current_idx = 0  # at first only display MAE on new observations
        return self.train

    def step(self, target):
        """
        given current index `t`
        `target_k` = `\hat{y}^{-h[k]}_{t+h[k]}`
        target is assumed to be a 1-d numpy array
        """
        # assign target to panda predicted values along diagonal
        assert target.shape == (len(self.h),)

        self.y_hat[self.h+self.current_idx,np.arange(len(self.h))] = target
        loss = MAE(self.test.loc[self.current_idx, 'y'],self.y_hat[self.current_idx,:])
        self.test.loc[self.current_idx, 'loss'] = loss
        current = self.test.ix[self.current_idx].drop(['loss']).to_frame().transpose()\
            # new observation
        self.current_idx += 1
        if self.current_idx > (self.n_test-(self.h[-1])):
            done = True
            # the total score is average MAE over the test set. Since each perdiction is for the
            # same number of time steps
            score = (self.test.apply(lambda r: r.y-r.y_hat,axis=1).abs().sum(1).sum()/
                     (self.n_predictions*len(self.h)))
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
