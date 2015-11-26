"""
Makes this pipeline safe for in-place manipulations on training X and y, e.g. transformation & deletion.

"""
# Author: Edouard Duchesnay
#         Gael Varoquaux
#         Virgile Fritsch
#         Alexandre Gramfort
#         Lars Buitinck
# Licence: BSD

from collections import defaultdict
from warnings import warn

import numpy as np
from scipy import sparse

# Changed relative addressing to absolute addressing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import six
from sklearn.utils import tosequence
from sklearn.utils.metaestimators import if_delegate_has_method

__all__ = ['Pipeline']


class Pipeline(BaseEstimator):
    """Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.
    The final estimator only needs to implement fit.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.

    Read more in the :ref:`User Guide <pipeline>`.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    Attributes
    ----------
    named_steps : dict
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    Examples
    ---------
    """

    # BaseEstimator interface

    def __init__(self, steps):
        names, estimators = zip(*steps)
        if len(dict(steps)) != len(steps):
            raise ValueError("Provided step names are not unique: %s" % (names,))

        # shallow copy of steps
        self.steps = tosequence(steps)
        transforms = estimators[:-1]
        estimator = estimators[-1]

        for t in transforms:
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All intermediate steps of the chain should "
                                "be transforms and implement fit and transform"
                                " '%s' (type %s) doesn't)" % (t, type(t)))

        if not hasattr(estimator, "fit"):
            raise TypeError("Last step of chain should implement fit "
                            "'%s' (type %s) doesn't)"
                            % (estimator, type(estimator)))

    @property
    def _estimator_type(self):
        return self.steps[-1][1]._estimator_type

    def get_params(self, deep=True):
        if not deep:
            return super(Pipeline, self).get_params(deep=False)
        else:
            out = self.named_steps
            for name, step in six.iteritems(self.named_steps):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            out.update(super(Pipeline, self).get_params(deep=False))
            return out

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def _final_estimator(self):
        return self.steps[-1][1]

    # Estimator interface

    def _pre_transform(self, X, y=None, **fit_params):
        fit_params_steps = dict((step, {}) for step, _ in self.steps)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        # deep-copy data before pre-transform so that subsequent in-place
        # modifications by transformers are transparent to the caller. i.e. X, y is unchanged
        Xt = X.copy()
        yt = y.copy()
        for name, transform in self.steps[:-1]:
            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, yt, **fit_params_steps[name])
            else:
                Xt = transform.fit(Xt, yt, **fit_params_steps[name]) .transform(Xt)
        # also return yt for fitting estimator
        return Xt, yt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        """
        Xt, yt, fit_params = self._pre_transform(X, y, **fit_params)
        self.steps[-1][-1].fit(Xt, yt, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then use fit_transform on transformed data using the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        """
        Xt, yt, fit_params = self._pre_transform(X, y, **fit_params)
        if hasattr(self.steps[-1][-1], 'fit_transform'):
            return self.steps[-1][-1].fit_transform(Xt, yt, **fit_params)
        else:
            return self.steps[-1][-1].fit(Xt, yt, **fit_params).transform(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X):
        """Applies transforms to the data, and the predict method of the
        final estimator. Valid only if the final estimator implements
        predict.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in pipeline after transforms.

        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.
        """
        Xt, fit_params = self._pre_transform(X, y, **fit_params)
        return self.steps[-1][-1].fit_predict(Xt, y, **fit_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        """Applies transforms to the data, and the predict_proba method of the
        final estimator. Valid only if the final estimator implements
        predict_proba.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_proba(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X):
        """Applies transforms to the data, and the decision_function method of
        the final estimator. Valid only if the final estimator implements
        decision_function.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].decision_function(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_log_proba(self, X):
        """Applies transforms to the data, and the predict_log_proba method of
        the final estimator. Valid only if the final estimator implements
        predict_log_proba.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_log_proba(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def transform(self, X):
        """Applies transforms to the data, and the transform method of the
        final estimator. Valid only if the final estimator implements
        transform.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = X
        for name, transform in self.steps:
            Xt = transform.transform(Xt)
        return Xt

    @if_delegate_has_method(delegate='_final_estimator')
    def inverse_transform(self, X):
        """Applies inverse transform to the data.
        Starts with the last step of the pipeline and applies ``inverse_transform`` in
        inverse order of the pipeline steps.
        Valid only if all steps of the pipeline implement inverse_transform.

        Parameters
        ----------
        X : iterable
            Data to inverse transform. Must fulfill output requirements of the
            last step of the pipeline.
        """
        if X.ndim == 1:
            warn("From version 0.19, a 1d X will not be reshaped in"
                 " pipeline.inverse_transform any more.", FutureWarning)
            X = X[None, :]
        Xt = X
        for name, step in self.steps[::-1]:
            Xt = step.inverse_transform(Xt)
        return Xt

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None):
        """Applies transforms to the data, and the score method of the
        final estimator. Valid only if the final estimator implements
        score.

        Parameters
        ----------
        X : iterable
            Data to score. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all steps of
            the pipeline.
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].score(Xt, y)

    @property
    def classes_(self):
        return self.steps[-1][-1].classes_

    @property
    def _pairwise(self):
        # check if first estimator expects pairwise input
        return getattr(self.steps[0][1], '_pairwise', False)


class TransformerPipeline(BaseEstimator, TransformerMixin):
    """(Shared) Pipeline of transforms.

    Sequentially apply a list of transforms. Steps of the pipeline must be 'transforms', that is, they
    must implement fit and transform methods.

    Pipeline may be shared between multiple paths in the execution DAG to save memory and execution time.
    In such a case, Pipeline will only run fit, fit_transform, and transform every n calls, where
    n is the number of paths that share this pipeline, and pass-through will apply for the rest
    of calls

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained.

    num_shares : Integer
        Number of path that share this pipeline.

    Attributes
    ----------
    named_steps : dict
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    call_counter : Integer
        Number of calls to this pipeline. Decides when to re-fit.
    """

    def __init__(self, steps, num_shares=1):
        names, transforms = zip(*steps)
        if len(dict(steps)) != len(steps):
            raise ValueError("Provided step names are not unique: %s" % (names,))

        # shallow copy of steps
        self.steps = tosequence(steps)
        for t in transforms:
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All intermediate steps of the chain should "
                                "be transforms and implement fit and transform"
                                " '%s' (type %s) doesn't)" % (t, type(t)))
        # pipeline states
        self.num_shares = num_shares
        self.call_counter = 0

    @property
    def named_steps(self):
        return dict(self.steps)

    # Transformer interface
    def fit(self, X, y=None, **fit_params):
        """
        Dummy fit do nothing. fit_transform do all the stuff
        """
        print "Warning: calling the dummy fit in TransformerPipeline"
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then use fit_transform on transformed data using the final
        estimator.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        """
        if self.call_counter == 0:
            fit_params_steps = dict((step, {}) for step, _ in self.steps)
            for pname, pval in six.iteritems(fit_params):
                step, param = pname.split('__', 1)
                fit_params_steps[step][param] = pval
            # we want to allow tranformers to modify X and y in-place, so the following operations
            # are directly applied to X and y in-place
            for name, transform in self.steps:
                if hasattr(transform, "fit_transform"):
                    X = transform.fit_transform(X, y, **fit_params_steps[name])
                else:
                    X = transform.fit(X, y, **fit_params_steps[name]) .transform(X)
        self.call_counter = (self.call_counter+1) % self.num_shares
        return X

    def transform(self, X):
        """Applies transforms to the data, and the transform method of the
        final estimator. Valid only if the final estimator implements
        transform.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step of
            the pipeline.
        """
        Xt = X
        for name, transform in self.steps:
            Xt = transform.transform(Xt)
        return Xt

    def inverse_transform(self, X):
        """Applies inverse transform to the data.
        Starts with the last step of the pipeline and applies ``inverse_transform`` in
        inverse order of the pipeline steps.
        Valid only if all steps of the pipeline implement inverse_transform.

        Parameters
        ----------
        X : iterable
            Data to inverse transform. Must fulfill output requirements of the
            last step of the pipeline.
        """
        raise NotImplementedError
