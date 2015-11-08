__author__ = 'shangxing'

#Combine mutiple samples of the same Id into one single sample by using mean value instead
#train_10_processed: a DataFrame structure containg the processed data


import dask.dataframe as dd
import pandas as pd
import numpy as np
import sys
import pickle as df
from sklearn.preprocessing import Imputer



def func(df,x):


    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

    estimates = imp.fit_transform(df)

    index = estimates.shape

    if index[1] < 24: #remove those Ids which have some columns of all NAN
        return pd.DataFrame()

    else:
        return pd.DataFrame(estimates,columns=x)





temp = pd.read_csv('train_10.csv') #read data


temp2=temp.loc[:,'Id':'Expected']

labels = list(temp2)

a=temp2.groupby(temp2.Id).apply(lambda d: func(d,labels)) #replace Nan with mean values

b = a.reindex(columns = labels)

train_10_processed = b.groupby(b.Id).mean() # use mean value as the single observation for each Id


