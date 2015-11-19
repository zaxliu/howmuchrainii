from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd


class TargetThresholdFilter(TransformerMixin):
    def __init__(self, threshold=None):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Dummy fit do nothing. fit_transform do all the stuff
        """
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Modify X and y in-place for training set
        Although y is not returned to comply with sklearn API, subsequent transformers will see a different y
        """
        indices = X.index[(y > self.threshold).nonzero()[0]]  # since we are dealing DataFrame, the index we get must be the index of DataFrame     
        X.drop(indices, inplace=True)
        y.drop(indices, inplace=True)
        return X

    def transform(self, X, y=None):
        """
        Dummy function do nothing
        """
        return X


class LogPlusOne(TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        y.loc[:] = np.log10(1+y)
        return X

    def transform(self, X):
        return X

    def metric(self, ground_truth, predictions):
        return np.float64(
            np.mean(
                np.abs(
                    ground_truth - (np.power(10, predictions) - 1)
                )
            )
        )



