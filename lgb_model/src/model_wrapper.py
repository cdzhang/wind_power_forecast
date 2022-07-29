# -*-Encoding: utf-8 -*-
"""
Description: Model Training
Authors: aihongfeng
Date:    2022/05/20
Function:
    Model Training
    Model Validing
    Model Predicting
"""
import copy
import re
import os
import gc
import numpy as np
import pandas as pd
from typing import List, Dict
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.abspath(__file__).split('lgb_model')[0])
from lgb_model.src.prepare import prep_env
from lgb_model.src.wind_turbine_data import inverse_standardize, data_preprocess
import lgb_model.src.metrics as metrics


def read_dataset(data, data_type):
    df = data[data['data_type']==data_type]
    df = df.reset_index(drop=True)
    return df  


def eval_lgb_result(df):
    """Evaluate lgb prediciton"""
    # shift target and pred_target to align timstamp
    ts_IDs = df['TurbID'].unique()
    new_df = []
    for ts_ID in ts_IDs:
        a_df = df[df['TurbID']==ts_ID]
        a_df = a_df.sort_values(['TurbID', 'date'])
        a_df = a_df.shift(288)
        new_df.append(a_df)
    new_df = pd.concat(new_df, axis=0)
    raw_data_ls = [a_df for _, a_df in new_df.groupby('TurbID')]
    # prepare pred and true, format: (time number * turbine number, 288, 1)
    turbids = new_df['TurbID'].unique()
    pred = []
    true = []
    a_arr_len = []
    for turbid in turbids:
        a_arr = new_df[new_df['TurbID']==turbid][['target','pred_target']].values
        a_arr_len.append(len(a_arr))
        for i in range(len(a_arr)):
            true.append(list(a_arr[i:i+288, 0]))
            pred.append(list(a_arr[i:i+288, 1]))
            if i+288 == len(a_arr):
                break
    pred = np.array(pred)
    true = np.array(true)
    pred = pred.reshape(-1,288,1)
    true = true.reshape(-1,288,1)
    # evaluating
    print(f"========{df['data_type'].iloc[0]}========")
    metrics.KDD_Score(pred, true, raw_data_ls, s_ind=0, num_turb=134, stride=1)



def train(settings):
    """
    Offline Model Training, Evaluation, Prediction
    """
    ########################################
    # LightGBM
    ########################################
    # load dataset
    all_df = pd.read_csv(settings['preprocess_data_path'])
    all_df = all_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    train_df = read_dataset(all_df, 'train')
    valid_df = read_dataset(all_df, 'valid')
    test_df = read_dataset(all_df, 'test')
    print('Init | train_df:{}, valid_df:{}, test_df:{}'.format(train_df.shape, valid_df.shape, test_df.shape))
    del all_df

    # define the columns
    TARGET_COL = 'target'
    PRED_TARGET_COL = 'pred_' + TARGET_COL
    CAT_COL = ['TurbID']
    USELESS_COL = ['Day', 'Tmstamp', 'date', 'data_type', 'abnormal', 'time']
    USE_COL = [ c for c in list(train_df) if (c not in USELESS_COL) and (c != TARGET_COL)]
    print("USE_COL:", USE_COL)

    # create lgb dataset
    print('The Number of Using Features:', len(USE_COL))
    train_matrix = lgb.Dataset(train_df[USE_COL], label=train_df[TARGET_COL], categorical_feature=CAT_COL)
    valid_matrix = lgb.Dataset(valid_df[USE_COL], label=valid_df[TARGET_COL], categorical_feature=CAT_COL)
    test_matrix = lgb.Dataset(test_df[USE_COL], label=test_df[TARGET_COL], categorical_feature=CAT_COL)

    # define model parameter
    # note for model ensemble: we use seed=50,100,500,1000,2000 and colsample_bytree&subsample=0.6,0.8 
    model_params = {
    'objective':'regression',
    'metric':{'rmse'},
    'learning_rate':0.03,
    'seed':2022, # 50,100,500,1000,2000
    'boosting_type':'gbdt', # dart don't support early stopping
    'early_stopping_round':10,
    'colsample_bytree':0.6,# 0.8
    'subsample': 0.6,# 0.8
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'n_jobs': -1,
    'verbose':-1
    }   

    # train model
    lgb_model = lgb.train(model_params,
                            train_matrix,
                            num_boost_round=500,
                            valid_sets=[valid_matrix], 
                            verbose_eval=1)

    # predict
    valid_df.loc[:, PRED_TARGET_COL] = lgb_model.predict(valid_df[USE_COL], num_iteration=lgb_model.best_iteration)
    test_df.loc[:, PRED_TARGET_COL] = lgb_model.predict(test_df[USE_COL], num_iteration=lgb_model.best_iteration)  
    del train_matrix, valid_matrix, test_matrix

    # inverse standardization
    valid_df = inverse_standardize(valid_df, settings['scaler_save_path'], settings['is_submit'])
    test_df = inverse_standardize(test_df, settings['scaler_save_path'], settings['is_submit'])


    # do the short-term adjustment
    weights = []
    for t in range(288):
        # when to become 0
        zero_loc = 75
        if t < zero_loc:
            weights.append((t-zero_loc)**2/zero_loc**2)
        else:
            weights.append(0) 
    lookback_num = 10 # lookback window size
    dev_df = pd.concat([valid_df, test_df], axis=0).sort_values(['TurbID','Day','Tmstamp'], ascending=True)
    turbids = dev_df['TurbID'].unique()
    new_dev_df = []
    for turbid in turbids:
        a_df = dev_df[dev_df['TurbID']==turbid]
        a_df = a_df.reset_index(drop=True)
        for start_idx in np.arange(0, len(a_df), 288)[1:]:
            # compute the mean value of a lookback window of gt Patv
            gt_patv = copy.deepcopy(a_df['target'].iloc[start_idx-lookback_num:start_idx])
            gt_patv.loc[a_df['abnormal'].iloc[start_idx-lookback_num:start_idx]] = np.nan
            mean_gt_patv = np.nanmean(gt_patv)
            # compute the mean value of a lookback window of pred Patv
            pred_patv = copy.deepcopy(a_df['pred_target'].iloc[start_idx-lookback_num:start_idx])
            pred_patv.loc[a_df['abnormal'].iloc[start_idx-lookback_num:start_idx]] = np.nan
            mean_pred_patv = np.nanmean(pred_patv)
            # compute the abnormal percentage of the lookback window
            ab_pert = np.sum(a_df['abnormal'].iloc[start_idx-lookback_num:start_idx]) / lookback_num
            # weighted residual
            residual_patv = mean_gt_patv - mean_pred_patv
            weight_residual_patv = np.array(weights) * residual_patv
            # if low abnormal percentage, we do the short-term adjustment
            if ab_pert < 0.6:
                a_df['pred_target'].iloc[start_idx:start_idx+288] = mean_pred_patv + weight_residual_patv
        new_dev_df.append(a_df)
    new_dev_df = pd.concat(new_dev_df, axis=0)
    valid_df = new_dev_df[new_dev_df['data_type']=='valid'].reset_index(drop=True)
    test_df = new_dev_df[new_dev_df['data_type']=='test'].reset_index(drop=True)

    # evaluation
    eval_lgb_result(valid_df)
    eval_lgb_result(test_df)
    
    # save pred result
    # valid_df.to_csv('/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/src_gbdt/lgb_valid_res_df.csv', index=False)
    # test_df.to_csv('/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/src_gbdt/lgb_test_res_df.csv', index=False)
    
    # save the best model
    save_model_flag = True
    if save_model_flag == True:
        lgb_model.save_model(settings['lgb_model_save_path'], num_iteration=lgb_model.best_iteration)

    # plot feature importance figure
    import_df = pd.DataFrame()
    import_df['feature_name'] = USE_COL
    import_df['importance'] = lgb_model.feature_importance(iteration=lgb_model.best_iteration)
    import_df = import_df.sort_values(by='importance')
    print(import_df.sort_values(by='importance', ascending=False))
    fig = plt.figure(figsize=(20,40))
    plt.barh(import_df['feature_name'], import_df['importance'])
    plt.title('LightGBM: The feature importance')
    plt.savefig(settings['lgb_feat_import_save_path'])


class ModelWrapper():
    def __init__(self, settings, models):
        self.settings = settings
        self.models = models
        self.scaler_save_path = settings['scaler_save_path']

    def predict(self):
        """
        predict
        """
        # load data and preprocess
        all_df = data_preprocess(self.settings)
        pred_df = all_df[all_df['data_type']=='pred']

        # define the columns
        TARGET_COL = 'target'
        PRED_TARGET_COL = 'pred_' + TARGET_COL
        CAT_COL = ['TurbID']
        USELESS_COL = ['Day', 'Tmstamp', 'date', 'data_type', 'abnormal', 'time']
        USE_COL = [c for c in list(pred_df) if (c not in USELESS_COL) and (c != TARGET_COL)]
        
        # LGB prediction
        tmodel_num = len(self.models)
        t_ensemble_pred_df = pd.DataFrame()
        # forloop each model
        for i in range(tmodel_num):
            # predict
            pred_df.loc[:, PRED_TARGET_COL] = self.models[i].predict(pred_df[USE_COL], num_iteration=self.models[i].best_iteration)
            # inverse standardization
            t_pred_df = inverse_standardize(pred_df, self.scaler_save_path, self.settings['is_submit'])
            if len(t_ensemble_pred_df) == 0:
                t_ensemble_pred_df = t_pred_df
            else:
                t_ensemble_pred_df.loc[:, PRED_TARGET_COL] += t_pred_df[PRED_TARGET_COL].values
        # average prediction
        pred_df.loc[:, PRED_TARGET_COL] = t_ensemble_pred_df[PRED_TARGET_COL].values / tmodel_num
        return pred_df


def load_model(settings):        
    """load model"""
    tmodels = []
    tmodel_path = os.path.dirname(os.path.abspath(__file__))
    tmodel_path = os.path.join(tmodel_path, 'tmodels')
    for p in os.listdir(tmodel_path):
        model_path = os.path.join(tmodel_path, p)
        t_model = lgb.Booster(model_file=model_path)
        tmodels.append(t_model)
    return tmodels


def load_trained_model_wrapper(settings):
    """read pretrained model and settings"""
    return ModelWrapper(settings, load_model(settings))

