import numpy as np
import pandas as pd
from anrg.pipeline import Pipeline, TransformerPipeline
from anrg.cleaning import TargetThresholdFilter, LogPlusOne
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import FeatureUnion

X1 = pd.DataFrame(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))
X2 = pd.DataFrame(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))
y1 = pd.Series(np.array([1, 2, 3, 4]))
y2 = pd.Series(np.array([1, 2, 3, 4]))
pip = TransformerPipeline(steps=[('ttf', TargetThresholdFilter(threshold=3))], num_shares=2)
PIP = Pipeline([('pip', pip), ('clf', DummyRegressor(strategy='mean'))])
print '============'
print X1
print y1
print X2
print y2
print '============'
print PIP.fit(X1, y1).predict(X1)
print '============'
print PIP.fit(X2, y2).predict(X2)
print '============'
print PIP.fit(X2, y2).predict(X2)
print '============'
print PIP.fit(X2, y2).predict(X2)