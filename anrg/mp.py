import numbers
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_validation import _safe_split, KFold, StratifiedKFold, ShuffleSplit
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples


def check_cv(cv, X=None, y=None, classifier=False):
    """Input checker utility for building a CV in a user friendly way.
    Parameters
    ----------
    cv : int, double, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - double, to specify fraction of test set
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.
    X : array-like
        The data the cross-val object will be applied on.
    y : array-like
        The target variable for a supervised learning problem.
    classifier : boolean optional
        Whether the task is a classification task, in which case
        stratified KFold will be used.
    Returns
    -------
    checked_cv: a cross-validation generator instance.
        The return value is guaranteed to be a cv generator instance, whatever
        the input type.
    """
    is_sparse = sp.issparse(X)
    if cv is None:
        cv = 3
    if isinstance(cv, numbers.Integral):
        if classifier:
            if type_of_target(y) in ['binary', 'multiclass']:
                cv = StratifiedKFold(y, cv)
            else:
                cv = KFold(_num_samples(y), cv)
        else:
            if not is_sparse:
                n_samples = len(X)
            else:
                n_samples = X.shape[0]
            cv = KFold(n_samples, cv)
    elif isinstance(cv, float) and cv > 0 and cv < 1:
        if classifier:
            raise NotImplementedError
        else:
            if not is_sparse:
                n_samples = len(X)
            else:
                n_samples = X.shape[0]
            cv = ShuffleSplit(n=n_samples, test_size=cv, random_state=12345)
    return cv

class MPRegressor(BaseEstimator, RegressorMixin):
    """Marshall Palmar: a meta-regressor

    Using the Marshal Palmar equation with power scaling factor to be tuned.

    Parameters
    ----------
    power_scaling : double, default: 0.9
        MP's power scaling factor (Recommended range 0-1). Optimal "should" be around 0.85

    Attributes
    ----------
    power_scaling:
    """
    def __init__(self, power_scaling=0.9):
        self.power_scaling = power_scaling

    def fit(self, X, y=None):
        return self  # return-self convention

    def predict(self, X):
        mmperhr = pow(pow(10, X['Ref_mean']/10)/200, 0.625 * self.power_scaling)
        return mmperhr 

    def transform(self):
        # for pipeline
        pass

    def fit_transform(self, X, y=None):
        # for pipeline
        pass