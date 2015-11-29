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

class KDPZDRRegressor(BaseEstimator, RegressorMixin):
    """KDPZDR: a meta-regressor

    

    Parameters
    ----------
    kdpzdr_aa = 136
    kdpzdr_bb = 0.968
    kdpzdr_cc = -2.86
   
    Attributes
    ----------
    kdpzdr_aa_scaling:
    kdpzdr_bb_scaling:
    kdpzdr_bb_scaling:
    
    """
    def __init__(self, kdpzdr_aa_scaling=1,kdpzdr_bb_scaling=1,kdpzdr_cc_scaling=1):
        self.kdpzdr_aa_scaling = kdpzdr_aa_scaling
        self.kdpzdr_bb_scaling = kdpzdr_bb_scaling
        self.kdpzdr_cc_scaling = kdpzdr_cc_scaling

    def fit(self, X, y=None):
        return self  # return-self convention

    def predict(self, X):
        kdpzdr_aa = 136
        kdpzdr_bb = 0.968
        kdpzdr_cc = -2.86
        
        #mmperhr = np.sign(X['Kdp_mean'])*(kdpzdr_aa*self.kdpzdr_aa_scaling)*pow(np.abs(X['Kdp_mean']),kdpzdr_bb*self.kdpzdr_bb_scaling)
        
        mmperhr = np.sign(X['Kdp_mean'])*(kdpzdr_aa*self.kdpzdr_aa_scaling)*pow(np.abs(X['Kdp_mean']),kdpzdr_bb*self.kdpzdr_bb_scaling)*pow(X['Zdr_mean'],kdpzdr_cc*self.kdpzdr_cc_scaling)
    
        mmperhr[mmperhr <= 0]=0
        
        return mmperhr 

    def transform(self):
        # for pipeline
        pass

    def fit_transform(self, X, y=None):
        # for pipeline
        pass