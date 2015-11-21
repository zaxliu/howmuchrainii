'''
examples of using xgboost sklearn wrapper with early stopping. 

@author Devin
'''
import pandas as pd
import numpy as np 
from sklearn import preprocessing
import sys
sys.path.append('xgboost-master/wrapper')
import xgboost as xgb
from sklearn import svm, ensemble, linear_model, cross_validation
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer

def get_X_trning_data(X_trning_path):
    trn_all = pd.read_csv(X_trning_path)
    index=list(trn_all)
    #my_indices = [0,1,2,3,5,6,7,9,10,15,17, 18, 23]
    #my_indices = [0,1,2,3,7,11,15,19,23]
    my_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    trn_new = trn_all[[index[i] for i in my_indices]]
    trn_new = trn_new[trn_new['Expected']<69]

    #combine observations with same ID by using mean
    trn_mean = trn_new.groupby(trn_new.Id).agg(['mean', 'median', 'std', 'count','min','max'])
    trn_mean.columns = ['_'.join(col).strip() for col in trn_mean.columns.values]
    print(trn_mean)
    # ignore id's where all Ref vales are NaN
    trn_mean = trn_mean[pd.notnull(trn_mean.Ref_mean)]

    # replace missing values by mean
    index2 = list(trn_mean)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    trn_mean= pd.DataFrame(imp.fit_transform(trn_mean),index = trn_mean.index, columns=index2)

    #X_trn and test data preparation
    y_trn = np.log1p(trn_mean.loc[:,'Expected_mean'].values)
    #X_trn = trn_mean.loc[:,'minutes_past_mean':'Zdr_5x5_90th_count'].values
    #X_trn = trn_mean.loc[:,'minutes_past_mean':'Kdp_count'].values
    X_trn = trn_mean.loc[:,'minutes_past_mean':'Kdp_5x5_90th_max'].values

    #print(X_trn)

    return X_trn, y_trn


print('Loading X_trning data')
X_trn, y_trn = get_X_trning_data('../data/train.csv')

# X_trn = np.array(X_trn)
# test = np.array(test)

# # label encode the categorical variables
# for i in range(X_trn.shape[1]):
#     if type(X_trn[1,i]) is str:
#         lbl = preprocessing.LabelEncoder()
#         lbl.fit(list(X_trn[:,i]) + list(test[:,i]))
#         X_trn[:,i] = lbl.transform(X_trn[:,i])
#         # test[:,i] = lbl.transform(test[:,i])
#
# X_trn = X_trn.astype(float)
# # test = test.astype(float)

#sklearn grid search with xgboost early stopping. 
param_grid = {'learning_rate': [0.05,0.01],'max_depth':[5,7]}
boost = xgb.XGBRegressor(n_estimators=2000,early_stopping_rounds=5,early_stopping_perc=0.1,nthread=2)

gridsearch = GridSearchCV(boost, param_grid,cv=4, n_jobs=4)
gridsearch.fit(X_trn,y_trn)

## sample weights, and classification 
sample_weights = np.random.random(len(y_trn))

# create a binary classification task out of the y_trn
y_trn = y_trn.replace(range(2,100), 0)

# check early stopping for the classifier
# boost = xgb.XGBClassifier(n_estimators=2000,early_stopping_rounds=5,early_stopping_perc=0.1,nthread=1)
# boost.fit(X_trn,y_trn)

# check that using sample_weight will still train the classifier.
# boost.fit(X_trn,y_trn, sample_weights)

#check classifier without early stopping 
boost = xgb.XGBClassifier(n_estimators=20,nthread=1)
boost.fit(X_trn,y_trn)


print('OK')