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

class KDPRegressor(BaseEstimator, RegressorMixin):
    """KDP: a meta-regressor

    

    Parameters
    ----------
    kdp_aa = 40.6 #default
    kdp_bb = 0.866 #default
   
    Attributes
    ----------
    kdp_aa_scaling:
    kdp_bb_scaling:
    """
    def __init__(self, kdp_aa_scaling=1,kdp_bb_scaling=1):
        self.kdp_aa_scaling = kdp_aa_scaling
        self.kdp_bb_scaling = kdp_bb_scaling

    def fit(self, X, y=None):
        return self  # return-self convention

    def predict(self, X):
        kdp_aa = 4.06
        kdp_bb = 0.0866
        mmperhr = np.sign(X['Kdp_mean'])*(kdp_aa*self.kdp_aa_scaling)*pow(np.abs(X['Kdp_mean']),kdp_bb*self.kdp_bb_scaling)
        
        mmperhr[mmperhr <= 0]=0
           
       
        return mmperhr 

    def transform(self):
        # for pipeline
        pass

    def fit_transform(self, X, y=None):
        # for pipeline
        pass