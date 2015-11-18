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
        indices = np.nonzero(y > self.threshold)[0]
        X.drop(indices, inplace=True)
        y.drop(indices, inplace=True)
        return X

    def transform(self, X, y=None):
        """
        Dummy function do nothing
        """
        return X

