import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.pipeline import Pipeline
from blending import BlendedRegressor
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import load_boston
from sklearn.cross_validation import cross_val_score

boston = load_boston()
X = boston.data
y = boston.target

scores = cross_val_score(estimator=RandomForestRegressor(n_estimators=40), X=X, y=y, scoring='mean_squared_error', cv=10)
print scores
print np.mean(scores), np.std(scores)
#
# scores = cross_val_score(estimator=BlendedRegressor((RandomForestRegressor(n_estimators=40), LinearRegression(), LinearSVR()),
#                         blending_model=LinearRegression(), blending_split=10),
#                          X=X, y=y, scoring='mean_squared_error', cv=10)
# print scores
# print np.mean(scores), np.std(scores)

# scores = cross_val_score(estimator=BlendedRegressor((RandomForestRegressor(n_estimators=40), LinearRegression(), LinearSVR()),
#                         blending_model=LinearRegression(), blending_split=0.1),
#                          X=X, y=y, scoring='mean_squared_error', cv=10)
# print scores
# print np.mean(scores), np.std(scores)

scores = cross_val_score(estimator=BlendedRegressor((RandomForestRegressor(n_estimators=40), LinearRegression(), LinearSVR()),
                        blending_model=RandomForestRegressor(), blending_split=0.1, with_feature=True),
                         X=X, y=y, scoring='mean_squared_error', cv=10)
print scores
print np.mean(scores), np.std(scores)