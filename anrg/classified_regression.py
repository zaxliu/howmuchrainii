import numbers
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyClassifier

class ClassifiedRegressor(BaseEstimator, RegressorMixin):
    """Use classifier to choose which regressor to use

    Classify samples into multiple class and do regressor for each class separately. User can
    the labeling metric for real input, the classification model, and the regressors.

    Parameters
    ----------
    labeling_thresh : number, default: None
        Threshold for labeling samples
    classifier : classifier object, default: None
        How to split samples
    proba_thresh : function, default: None
        Threshold for deriving class label through probability
    Regressors : tuple, default: (LinearRegression(),)
        Regressors to use for each class. Length should be greater or equal to number of classes

    Attributes
    ----------

    """
    def __init__(self, labeling_thresh=None,
                 classifier=None, proba_thresh=None,
                 regressors=(LinearRegression(), )):
        if labeling_thresh is None or classifier is None or proba_thresh is None:
            if not labeling_thresh is None or not classifier is None or not proba_thresh is None:
                print "labeling_metric, classifier, proba_to_label should be None simultaneously!"
                raise ValueError
            else:
                self.labeling_thresh = np.inf
                self.classifier = DummyClassifier(strategy='constant', constant=0)
                self.proba_thresh = 0.5
        else:
            self.labeling_thresh = labeling_thresh
            self.classifier = classifier
            self.proba_thresh = proba_thresh
        self.regressors = regressors

    def fit(self, X, y=None):
        # step 1: (X, y) -> (X, label)
        print "Getting labels..."
        labels = y > self.labeling_thresh
        # step 2: (X, label) train classifier
        # print "Training classifier..."
        self.classifier.fit(X, labels)
        # step 3: (X[label=i], y[label=i]) train regressors
        # print "Training regressors...",
        label_set = np.unique(labels)  # get the set of unique labels, in ascending sorted order
        # Sanity check
        if len(label_set) != len(self.regressors):
            print "Number of labels != number of regressors!"
            raise ValueError
        for i, label in enumerate(label_set):
            # print " {}...".format(i),
            index = (labels == label)
            self.regressors[i].fit(X[index, :], y[index])
        # print " "
        return self

    def predict(self, X):
        # Step 1: X -> proba
        probs = self.classifier.predict_proba(X)
        # Step 2: proba -> label
        labels = probs[:, 0] < self.proba_thresh
        # Step 3: X[label=i] -> regressor prediction
        label_set = np.unique(labels)
        y = np.zeros([X.shape[0], ])
        for i, label in enumerate(label_set):
            index = (labels == label)
            y[index] = self.regressors[i].predict(X[index, :])
        return y

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)




