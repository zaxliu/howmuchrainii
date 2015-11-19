from cleaning import TargetThresholdFilter, LogPlusOne
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# # Check basic functionality
# # only works for pandas DataFrame and Series because we are modifying shape in-place
# X1 = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
# X2 = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
# y1 = pd.Series(np.array([1, 2, 3, 4]))
# y2 = pd.Series(np.array([1, 2, 3, 4]))
# ttf = TargetThresholdFilter(threshold=3)
# ttf.fit_transform(X1, y1)
# ttf.transform(X2, y2)
# print X1
# print y1
# print X2
# print y2
#
# # sklearn pipe compatability
# print "==================="
# X1 = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
# X2 = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
# y1 = pd.Series(np.array([1, 2, 3, 4]))
# y2 = pd.Series(np.array([1, 2, 3, 4]))
# steps = [('ttf', TargetThresholdFilter(threshold=1)), ('lr', LinearRegression())]
# pip = Pipeline(steps)
# pip.fit(X1, y1)
# print 'X1'
# print X1
# print 'y1'
# print y1
# print 'X2'
# print X2
# print 'predict2'
# print pip.predict(X2)

# log(1+y)
X1 = pd.DataFrame(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
y1 = pd.Series(np.array([1, 2, 3, 4]))
y2 = pd.Series(np.array([1, 2, 3, 4]))
lpo = LogPlusOne()
lpo.fit_transform(X1, y1)

print X1
print y1
print lpo.transform(X1)
print lpo.metric(y2, y1)
