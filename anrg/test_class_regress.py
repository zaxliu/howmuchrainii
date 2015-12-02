import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from classified_regression import ClassifiedRegressor

boston = load_boston()
X = boston.data
y = boston.target

# Dummy case, should be equal to Linear regression
cr = ClassifiedRegressor()
lr = LinearRegression()
scores1 = cross_val_score(cr, X, y, scoring='mean_absolute_error', cv=10, verbose=3)
scores2 = cross_val_score(lr, X, y, scoring='mean_absolute_error', cv=10, verbose=3)
print np.mean(scores1), np.std(scores1)
print np.mean(scores2), np.std(scores2)

# Median splitting case
cr = ClassifiedRegressor(labeling_thresh=np.median(y), classifier=RandomForestClassifier(n_estimators=10), proba_thresh=0.5,
                         regressors=(LinearRegression(), LinearRegression()))
lr = LinearRegression()
scores1 = cross_val_score(cr, X, y, scoring='mean_absolute_error', cv=10, verbose=3)
scores2 = cross_val_score(lr, X, y, scoring='mean_absolute_error', cv=10, verbose=3)
print np.mean(scores1), np.std(scores1)
print np.mean(scores2), np.std(scores2)
