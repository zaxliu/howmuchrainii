import numbers
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
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

class BlendedRegressor(BaseEstimator, RegressorMixin):
    """Blending multiple regressors into a meta-regressor

    Blending the output of multiple regressors into a meta-regressor using a blending model. User can
    specify how training data is splitted between base-training and blending-training phase.

    Parameters
    ----------
    base_models : tuple, default: LinearRegression()
        Tuple of base model objects.
    blending_model : object, default: LinearRegression()
        The model used for blending base models
    blending_split : None, int, double (0 to 1) or object, default: None
        The splitting method used for base- and blending training
    with_feature : Boolean, default: False
        Whether or not to use original features when learning blending model

    Attributes
    ----------
    base_models:
    blending_model:
    blending_split:
    with_feature:
    """
    def __init__(self, base_models=(LinearRegression(),), blending_model=LinearRegression(),
                 blending_split=None, with_feature=False):
        self.base_models = base_models
        self.blending_model = blending_model
        self.blending_split = blending_split
        self.with_feature = with_feature

    def fit(self, X, y=None):
        blending_split = check_cv(self.blending_split, X, y)  # return a cv iterator
        base_index, blending_index = blending_split.__iter__().next()  # take first split
        # fit base_models on X_base_trn and y_base_trn
        # print "Fitting base models..."
        for bm in self.base_models:
            X_base, y_base = _safe_split(bm, X, y, base_index)
            bm.fit(X_base, y_base)

        # fit blending_model on X_blending_trn and y_blending_trn
        # print "Fitting the blending model"
        X_base_blending, y_blending = _safe_split(self.blending_model, X, y, blending_index, base_index)
        if self.with_feature:
            # shape = [n_blending_samples, n_base_models+m_original_features]
            X_blending = np.zeros([y_blending.shape[0], X_base_blending.shape[1]+len(self.base_models)])
            X_blending[:, len(self.base_models):] = X_base_blending
        else:
            # shape=[n_blending_samples, n_base_models]
            X_blending = np.zeros([y_blending.shape[0], len(self.base_models)])
        for i, bm in enumerate(self.base_models):
            X_blending[:, i] = bm.predict(X_base_blending)
        self.blending_model.fit(X_blending, y_blending)
        return self  # return-self convention

    def predict(self, X):
        if self.with_feature:
            # shape = [n_blending_samples, n_base_models+m_original_features]
            X_blending = np.zeros([X.shape[0], X.shape[1]+len(self.base_models)])
            X_blending[:, len(self.base_models):] = X
        else:
            # shape=[n_blending_samples, n_base_models]
            X_blending = np.zeros([X.shape[0], len(self.base_models)])
        for i, bm in enumerate(self.base_models):
            X_blending[:, i] = bm.predict(X)
        return self.blending_model.predict(X_blending)

    def transform(self):
        # for pipeline
        pass

    def fit_transform(self, X, y=None):
        # for pipeline
        pass