import numpy as np
import pandas as pd
from anrg.pipeline import Pipeline, SelectTailK, LeaveTailK, DummyRegressor
from anrg.cleaning import TargetThresholdFilter, LogPlusOne
from sklearn.pipeline import FeatureUnion

X = np.array([[1,2,3,3], [1,2,3,4], [1,2,3,4], [1,2,3,4]])
y = np.array([1,2,3,4])

lt = LeaveTailK(K=3)
st = SelectTailK(K=3)
dr = DummyRegressor()

# print lt.fit_transform(X, y)
# print st.fit_transform(X, y)

dr.fit(X, y)
print dr.predict(X)