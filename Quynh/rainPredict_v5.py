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
    print(trn_mean)
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

    #print(X_trn)

    return X_trn, y_trn

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
    unique_ID = pd.unique(test_all['Id'])

    return test_X,test_mean,unique_ID

def xgb_cv(X_trn, y_trn, objective, num_folds, num_threads, num_round, eta, max_depth, min_child_weight,max_delta_step, gamma,subsample,colsample):


    param = {}
    param['silent'] = 1
    param['nthread'] = num_threads
    param['bst:max_depth'] = max_depth
    param['bst:eta'] = eta
    param['objective'] = objective
    param['min_child_weight'] = min_child_weight
    param['gamma'] = gamma
    param['max_delta_step'] = max_delta_step
    param['subsample'] = subsample
    param['colsample_bytree'] = colsample

    '''
    Cross validation for XGBoost performance with MAE
    '''

    best_model = None
    kf = cv.KFold(len(y_trn), n_folds=num_folds)
    best_cv = 1000000

    for train_indices, test_indices in kf:

        xgb_train_cv, xgb_test_cv = X_trn[train_indices], X_trn[test_indices]
        y_train_cv, y_test_cv = y_trn[train_indices], y_trn[test_indices]

        xgmat = xgb.DMatrix(xgb_train_cv, label=y_train_cv)
        plst = param.items()
        watchlist = []#[(xgmat, 'train')] #evallist  = [(dtest,'eval'), (dtrain,'train')]

        t = time.time()
        bst = xgb.train(plst, xgmat, num_round, watchlist)
        print(time.time()-t)

        xgmat_test = xgb.DMatrix(xgb_test_cv)
        tmp_predict = bst.predict(xgmat_test)
        y_predict = np.exp(tmp_predict)-1
        accu = MAE(y_predict,y_test_cv)

        if accu <= best_cv:
            best_model = bst

    file_name =  "rgb_nt_%d.model" % num_round
    best_model.dump_model(file_name)
    return best_model

def export_submission(unique_ID,prediction,processed_test,out_file):

    marshall = pd.read_csv('../data/MP_r_09.csv')
    test_result_exist = pd.DataFrame()
    test_result_exist['Id'] = processed_test.index
    test_result_exist['Expected'] = prediction

    test_result = pd.DataFrame()
    test_result['Id'] = unique_ID
    test_result = pd.merge(test_result, test_result_exist, how='left', on=['Id'], sort=True)
    test_result.loc[test_result['Expected'].isnull(), 'Expected'] = marshall.loc[test_result['Expected'].isnull(), 'Expected']
    test_result.loc[test_result['Expected'].notnull(), 'Expected'] = 0.75*test_result.loc[test_result['Expected'].notnull(), 'Expected']+0.25*marshall.loc[test_result['Expected'].notnull(), 'Expected']

    test_result.to_csv(out_file, index=False)

def main():

    # setup parameters for xgboost

    #XGB default
    #eta [default=0.3]: step size shrinkage
    #gamma [default=0]: minimum loss reduction
    #max_depth [default=6]
    #min_child_weight [default=1]
    #max_delta_step [default=0]
    #subsample [default=1]
    #colsample_bytree [default=1]
    #lambda [default=1]: L2 regularization
    #alpha [default=0]: L1 regularization

    objective = 'reg:linear'
    num_round = 200 # Number of boosted trees

    num_folds = 3
    num_threads = 4
    eta = 0.3
    max_depth = 5
    min_child_weight = 1
    max_delta_step = 0
    gamma = 0
    subsample = 1
    colsample = 1

    print('Loading training data')
    X_trn, y_trn = get_training_data('../data/train.csv')

    print('Training data with cross validation on Max Absolute Error')
    best_valid_model = xgb_cv(X_trn, y_trn, objective, num_folds, num_threads, num_round, eta, max_depth, min_child_weight, max_delta_step,gamma,subsample ,colsample)

    print('Prediction with the best validation model')
    X_test,processed_test,unique_ID  = get_testing_data('../data/test.csv')
    xgmat_test = xgb.DMatrix(X_test)
    tmp_predict = best_valid_model.predict(xgmat_test)
    xgb_predict = np.exp(tmp_predict)-1

    # print(len(xgb_predict))
    # print(len(unique_ID))

    print('Writing the submission file')
    out_file = '../data/xgb_result_v5_pure_allFeatures_2.csv'
    export_submission(unique_ID,xgb_predict,processed_test, out_file)


if __name__ == "__main__":
    main()