
# coding: utf-8

# In[3]:

import time
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from math import *
from sklearn import metrics
#
#pd.set_option('display.max_columns', 500)  # force pandas to display all columns for better visual inspection
# plot plots inline
#%matplotlib inline 


# In[4]:

trn_all = pd.read_csv('../data/train.csv')  # column #0 in our file is index
# trn_1 = pd.read_csv('../data/train_1.csv', index_col=0)


# In[5]:

#selected features and Cut off outliers of Expected >= 69
trn_new = trn_all[trn_all['Expected']<69]

del trn_all

# In[6]:

#trn_new


# In[7]:

#combine observations with same ID by using mean
#replace Nan by overall mean
trn_mean = trn_new.groupby(trn_new.Id).agg(['mean', 'median', 'std', 'count', 'sem', 'skew', 'min', 'max'])
trn_mean.columns = ['_'.join(col).strip() for col in trn_mean.columns.values]
#trn_mean = trn_mean.drop(['Expected_count', 'Expected_median', 'Expected_std', 'Expected_min', 'Expected_max'], axis =1)

del trn_new

# In[8]:

# ignore id's where all Ref vales are NaN
trn_mean = trn_mean[pd.notnull(trn_mean.Ref_mean)]


# In[9]:

index2 = list(trn_mean)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

trn_mean= pd.DataFrame(imp.fit_transform(trn_mean),index = trn_mean.index, columns=index2)


# In[10]:

#trn_mean


# In[11]:

#train and test data preparation
X_trn = trn_mean.loc[:,'minutes_past_mean':'Kdp_5x5_90th_max'].values
y_trn = np.log1p(trn_mean.loc[:,'Expected_mean'].values)

del trn_mean

# In[12]:

#X_trn


# In[13]:

clf = RandomForestRegressor(n_estimators=200,n_jobs=7)  # NOTE: n_jobs=-1 will use all of your cores, set to a prefered number


# In[14]:

#train model
t = time.time()
clf.fit(X_trn, y_trn)
print time.time()-t


# In[15]:

#generate test result
test_new = pd.read_csv('../data/test.csv')

#combine observations with same ID by using mean
#replace Nan by overall mean
test_mean = test_new.groupby(test_new.Id).agg(['mean', 'median', 'std', 'count', 'sem', 'skew', 'min', 'max'])
test_mean.columns = ['_'.join(col).strip() for col in test_mean.columns.values]

index2 = list(test_mean)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
test_mean= pd.DataFrame(imp.fit_transform(test_mean),index=test_mean.index,columns=index2)

test_X =test_mean.loc[:,'minutes_past_mean':'Kdp_5x5_90th_max'].values

test_y_predict = np.exp(clf.predict(test_X))-1

del test_new
del test_mean
del test_X

# In[17]:

#generate output file
#0.75 from prediction and 0.25 from marshall palmer
marshall = pd.read_csv('../data/MP_r_09.csv')

test_result_exist = pd.DataFrame()
test_result_exist['Id'] = test_mean.index
test_result_exist['Expected'] = test_y_predict

test_result = pd.DataFrame()
test_result['Id'] = test_new['Id'].unique()
test_result = pd.merge(test_result, test_result_exist, how='left', on=['Id'], sort=True)
test_result.loc[test_result['Expected'].isnull(), 'Expected'] = marshall.loc[test_result['Expected'].isnull(), 'Expected']
test_result.loc[test_result['Expected'].notnull(), 'Expected'] = 0.75*test_result.loc[test_result['Expected'].notnull(), 'Expected']+0.25*marshall.loc[test_result['Expected'].notnull(), 'Expected']

test_result.to_csv('../data/pks_result_v3_2611.csv', index=False)


# In[19]:

#test_result


# In[ ]:



