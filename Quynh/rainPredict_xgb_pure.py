import numpy as np
import time
import xgboost as xgb
import pandas as pd
import sklearn.cross_validation as cv
from sklearn.preprocessing import Imputer

def MAE(predicted,actual):
    AE = np.sum(np.abs(actual-predicted))
    MAE = AE/len(predicted)
    return MAE

def get_training_data(training_path):
    trn_all = pd.read_csv(training_path)
    index=list(trn_all)
    #my_indices = [0,1,2,3,5,6,7,9,10,15,17, 18, 23]
    #my_indices = [0,1,2,3,7,11,15,19,23]
    my_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    trn_new = trn_all[[index[i] for i in my_indices]]
    trn_new = trn_new[trn_new['Expected']<69]

    #combine observations with same ID by using mean
    trn_mean = trn_new.groupby(trn_new.Id).agg(['mean', 'median', 'std', 'count','min','max'])
    trn_mean.columns = ['_'.join(col).strip() for col in trn_mean.columns.values]
    # ignore id's where all Ref vales are NaN
    trn_mean = trn_mean[pd.notnull(trn_mean.Ref_mean)]

    # replace missing values by mean
    index2 = list(trn_mean)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    trn_mean= pd.DataFrame(imp.fit_transform(trn_mean),index = trn_mean.index, columns=index2)

    #train and test data preparation
    y_trn = np.log1p(trn_mean.loc[:,'Expected_mean'].values)
    #X_trn = trn_mean.loc[:,'minutes_past_mean':'Zdr_5x5_90th_count'].values
    #X_trn = trn_mean.loc[:,'minutes_past_mean':'Kdp_count'].values
    X_trn = trn_mean.loc[:,'minutes_past_mean':'Kdp_5x5_90th_max'].values
    unique_ID = trn_mean.index.tolist()
    return X_trn, y_trn,unique_ID

def get_testing_data(testing_path):
    test_all = pd.read_csv(testing_path)
    index=list(test_all)
    # my_indices = [0,1,2,3,5,6,7,9,10,15,17, 18]
    #my_indices = [0,1,2,3,7,11,15,19]
    my_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    test_new = test_all[[index[i] for i in my_indices]]
    test_mean = test_new.groupby(test_new.Id).agg(['mean', 'median', 'std', 'count','min','max'])
    test_mean.columns = ['_'.join(col).strip() for col in test_mean.columns.values]

    # Imputing with mean values
    index2 = list(test_mean)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    test_mean= pd.DataFrame(imp.fit_transform(test_mean),index=test_mean.index,columns=index2)
    #test_X = test_mean.loc[:,'minutes_past_mean':'Zdr_5x5_90th_count'].values
    #test_X = test_mean.loc[:,'minutes_past_mean':'Kdp_count'].values
    test_X = test_mean.loc[:,'minutes_past_mean':'Kdp_5x5_90th_max'].values
    unique_ID = test_mean.index.tolist()
    return test_X,unique_ID


def export_submission_pure(ID_list,prediction,out_file):

    test_result = pd.DataFrame()
    print(len(ID_list))
    print(len(prediction))
    test_result['Id'] = ID_list
    test_result['Expected'] = prediction
    test_result.to_csv(out_file, index=False)

def main():

    # setup parameters for xgboost

    num_round = 200 # Number of boosted trees
    param = {}
    param['silent'] = 1
    param['nthread'] = 4
    param['bst:max_depth'] = 5
    param['bst:eta'] = 0.3
    param['objective'] = 'reg:linear'
    param['min_child_weight'] = 1
    param['gamma'] = 0
    param['max_delta_step'] = 0
    param['subsample'] = 1
    param['colsample_bytree'] = 1

    print('Loading training data')
    X_trn, y_trn,trn_ID = get_training_data('../data/train.csv')


    print('Training data with Max Absolute Error')
    xgmat = xgb.DMatrix(X_trn, label=y_trn)
    plst = param.items()
    watchlist = []
    t = time.time()
    bst = xgb.train(plst, xgmat, num_round, watchlist)
    print(time.time()-t)

    print('Prediction for training data')
    tmp_predict = bst.predict(xgmat)
    trn_predict = np.exp(tmp_predict)-1

    print('Prediction for testing data')
    X_test,test_ID  = get_testing_data('../data/test.csv')
    xgmat_test = xgb.DMatrix(X_test)
    tmp_predict = bst.predict(xgmat_test)
    test_predict = np.exp(tmp_predict)-1

    print('Writing the submission file')
    out_file = '../data/xgb_result_v5_pure_allFeatures_train.csv'
    export_submission_pure(trn_ID,trn_predict,out_file)
    out_file = '../data/xgb_result_v5_pure_allFeatures_test.csv'
    export_submission_pure(test_ID,test_predict,out_file)

if __name__ == "__main__":
    main()