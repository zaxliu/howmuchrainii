import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# from sklearn.preprocessing import StandardScaler
# from sklearn.cross_validation import train_test_split
# from math import *
# from sklearn import metrics
# import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)  # force pandas to display all columns for better visual inspection

# plot plots inline
# %matplotlib inline

trn_all = pd.read_csv('../data/train.csv')  # column #0 in our file is index
# trn_1 = pd.read_csv('../data/train_1.csv', index_col=0)

#selected features and Cut off outliers of Expected >= 69
index=list(trn_all)
my_indices = [0,1,2,3,5,6,7,9,10,15,17, 18, 23]
trn_new = trn_all[[index[i] for i in my_indices]]

trn_new = trn_new[trn_new['Expected']<69]

trn_mean = trn_new.groupby(trn_new.Id).agg(['mean', 'median', 'std', 'count'])
trn_mean.columns = ['_'.join(col).strip() for col in trn_mean.columns.values]
index2 = list(trn_mean)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
trn_mean= pd.DataFrame(imp.fit_transform(trn_mean),index = trn_mean.index, columns=index2)
#train and test data preparation
X_trn = trn_mean.loc[:,'minutes_past_mean':'Zdr_5x5_90th_count'].values
y_trn = np.log1p(trn_mean.loc[:,'Expected_mean'].values)


# Split data as training and validation set
#[X_trn, X_test, y_trn, y_test] = train_test_split(X, Y, test_size = 0.3)

#clf = RandomForestRegressor(n_estimators=100,n_jobs=3)  # NOTE: n_jobs=-1 will use all of your cores, set to a prefered number

dtrain = xgb.DMatrix(X_trn,label=y_trn)
param = {'bst:max_depth':5, 'bst:eta':1, 'silent':1, 'objective':'reg:linear' }
param['nthread'] = 4
param['eval_metric'] = 'rmse'
plst = param.items()

evallist  = [(dtrain,'train')]


#train model
t = time.time()
num_round = 500
bst = xgb.train(plst, dtrain, num_round)
# bst = xgb.train(plst, dtrain, num_round, evallist)

#clf.fit(X_trn, y_trn)
print(time.time()-t)

bst.save_model('0001.model')

#generate test result
test = pd.read_csv('../data/test.csv')

#selected features and Cut off outliers of Expected >= 69
index=list(test)
my_indices = [0,1,2,3,5,6,7,9,10,15,17, 18]
test_new = test[[index[i] for i in my_indices]]
#Cut off Ref values < 0
#test_new[test_new['Ref_5x5_50th']<0] = np.nan
#test_new[test_new['Ref_5x5_90th']<0] = np.nan
#test_new[test_new['RefComposite']<0] = np.nan
#test_new[test_new['RefComposite_5x5_50th']<0] = np.nan
#test_new[test_new['RefComposite_5x5_90th']<0] = np.nan
#test_new[test_new['Ref']<0] = np.nan

#combine observations with same ID by using mean
#replace Nan by overall mean
test_mean = test_new.groupby(test_new.Id).agg(['mean', 'median', 'std', 'count'])
test_mean.columns = ['_'.join(col).strip() for col in test_mean.columns.values]

index2 = list(test_mean)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
test_mean= pd.DataFrame(imp.fit_transform(test_mean),index=test_mean.index,columns=index2)

test_X =test_mean.loc[:,'minutes_past_mean':'Zdr_5x5_90th_count'].values

dtest = xgb.DMatrix(test_X)

test_y_predict = np.exp(bst.predict(dtest))-1

#test_y_predict = np.exp(clf.predict(test_X))-1

#generate output file
#0.75 from prediction and 0.25 from marshall palmer
#marshall = pd.read_csv('../data/MP_r_09.csv')
marshall = pd.read_csv('../data/sample_solution.csv')

test_result_exist = pd.DataFrame()
test_result_exist['Id'] = test_mean.index
test_result_exist['Expected'] = test_y_predict

test_result = pd.DataFrame()
test_result['Id'] = test['Id'].unique()
test_result = pd.merge(test_result, test_result_exist, how='left', on=['Id'], sort=True)
test_result.loc[test_result['Expected'].isnull(), 'Expected'] = marshall.loc[test_result['Expected'].isnull(), 'Expected']
test_result.loc[test_result['Expected'].notnull(), 'Expected'] = 0.75*test_result.loc[test_result['Expected'].notnull(), 'Expected']+0.25*marshall.loc[test_result['Expected'].notnull(), 'Expected']

test_result.to_csv('../data/xgb_result_v4.csv', index=False)








