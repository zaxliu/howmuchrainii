import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import check_cv, _safe_split


class BlendedRegressor(BaseEstimator, RegressorMixin):
    """Blending multiple regressors into a meta-regressor

    Blending the output of multiple regressors into a meta-regressor using a blending model. User can
    specify how tranning data is splitted between base-training and blending-training phase.

    Parameters
    ----------
    base_models : tuple, default: LinearRegression()
        Tuple of base model objects.
    blending_model : object, default: LinearRegression()
        The model used for blending base models
    blending_split : None, int, or object, default: None
        The splitting method used for base- and blending training

    Attributes
    ----------
    base_models:
    blending_model:
    blending_split:
    """
    def __init__(self, base_models=(LinearRegression(),), blending_model=LinearRegression(), blending_split=None):
        self.base_models = base_models
        self.blending_model = blending_model
        self.blending_split = blending_split

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
        X_blending = np.zeros([y_blending.shape[0], len(self.base_models)])  # shape=[n_blending_samples, n_base_models]
        for i, bm in enumerate(self.base_models):
            X_blending[:, i] = bm.predict(X_base_blending)
        self.blending_model.fit(X_blending, y_blending)
        return self  # return-self convention

    def predict(self, X):
        X_blending = np.zeros([X.shape[0], len(self.base_models)])  # shape=[n_blending_samples, n_base_models]
        for i, bm in enumerate(self.base_models):
            X_blending[:, i] = bm.predict(X)
        return self.blending_model.predict(X_blending)

    def transform(self):
        # for pipeline
        pass

    def fit_transform(self, X, y=None):
        # for pipeline
        pass