# -*-Encoding: utf-8 -*-
"""
Description: Data Preparing
Authors: aihongfeng
Date:    2022/07/22
Function:
    Parameter and Data Loading
    Abnormal Preprocessing
    Feature Engineerging
    Label Generating
    Data Spliting
    Standardlizaiton
"""
import re
import os
import gc
import numpy as np
import pandas as pd
from typing import List, Dict
import lightgbm as lgb
from tqdm import tqdm
import copy 
from tsfresh import extract_features
import warnings 
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.abspath(__file__).split('lgb_model')[0])
from lgb_model.src.prepare import prep_env

def load_dataset(settings: dict):
    """Data Loading"""
    print('Data Reading.')
    if settings['is_submit'] == False:
        all_df = pd.read_csv(os.path.join(settings['data_path'], settings['filename']))
    else:
        all_df = pd.read_csv(settings['path_to_test_x'])
        # Online only use the recent total_size number of data, to reduce the preprocessing time cost
        all_df = all_df.iloc[-(settings['total_size']*134):,:] 
    all_df['date'] = [t for t in zip(all_df['Day'].values, all_df['Tmstamp'].values)]
    return all_df


def abnormal_preprocess(data: pd.DataFrame, settings: dict):
    """Abnormal Preprocessing"""
    print('Abnormal Preprocessing.')
    ##############################
    # Median Filling：Etmp
    ##############################
    # For the given time t, use 3-sigma rule apply to the whole field to find the anomalies
    # then use the median value of the filed at the time t to replace the abnormal values
    print('- 3gima: Etmp', )
    f = 'Etmp'
    # calulate the mean\std\median of each date
    tmp_data = data[['date','TurbID','Tmstamp','Day',f]]
    mu = tmp_data.groupby('date')[f].mean().reset_index().rename({f:'mean'}, axis=1)
    std = tmp_data.groupby('date')[f].std().reset_index().rename({f:'std'}, axis=1)
    median = tmp_data.groupby('date')[f].median().reset_index().rename({f:'median'}, axis=1)
    f_limit_df = pd.merge(mu, std, how='left', left_on='date', right_on='date')
    f_limit_df = pd.merge(f_limit_df, median, how='left', left_on='date', right_on='date')
    # calulate the limit of 3-sigma rule
    f_limit_df['bottom_limit'] = f_limit_df['mean'] - 3*f_limit_df['std']
    f_limit_df['upper_limit'] = f_limit_df['mean'] + 3*f_limit_df['std']
    # median filling
    data = pd.merge(data, f_limit_df[['date','bottom_limit','upper_limit', 'median']], how='left', left_on='date', right_on='date')
    data[f'abnormal_{f}'] = (data[f] < data['bottom_limit']) | (data[f] > data['upper_limit'])
    data.loc[data[f'abnormal_{f}'], f] = 0
    data[f] = data[f'abnormal_{f}']*data['median'] + data[f]
    data = data.drop(['bottom_limit','upper_limit','median'], axis=1)

    ##############################
    # Define the abnormal Patv
    ##############################
    # Since the first row of each turbine has NAN, we fill with 0
    data = data.fillna(0)
    # Define the abnormal Patv (not include Patv < 0)
    data['abnormal_patv'] = (data['Patv'] <= 0) & (data['Wspd'] > 2.5) | \
                        (data['Pab1'] > 89) | (data['Pab2'] > 89) | (data['Pab3'] > 89) | \
                        (data['Wdir'] < -180) | (data['Wdir'] > 180) | (data['Ndir'] < -720) | (data['Ndir'] > 720)
    # The abnormal Patv standard which uses for evaluation
    data['abnormal'] = (data['Patv'] < 0) | \
                        (data['Patv'] <= 0) & (data['Wspd'] > 2.5) | \
                        (data['Pab1'] > 89) | (data['Pab2'] > 89) | (data['Pab3'] > 89) | \
                        (data['Wdir'] < -180) | (data['Wdir'] > 180) | (data['Ndir'] < -720) | (data['Ndir'] > 720)

    ################################
    # Remove Feature：Pab1/2/3
    ################################
    # Their missing percentages are high, drop them
    print('- drop: Pab1,Pab2,Pab3')
    data = data.drop(['Pab1','Pab2','Pab3'], axis=1)

    # ################################
    # # Interpolate：Ndir
    # ################################
    data['abnormal_Ndir'] = (data['Ndir'] < -720) | (data['Ndir'] > 720)
    interp_feats = ['Ndir']
    for interp_feat in interp_feats:
        print('- Interpolate: ', interp_feat)
        data.loc[data[f'abnormal_{interp_feat}'], interp_feat] = np.nan
        # Interpolate for each turbine
        ab_turbids = data[data[f'abnormal_{interp_feat}']]['TurbID'].unique()
        new_data = []
        for turbid, a_data in data.groupby('TurbID'):
            if turbid not in ab_turbids:
                new_data.append(a_data) 
            else:
                # Interpolate first, then  back/forward fill to handle with the missing value on the first or bottom row
                a_data[interp_feat] = a_data[interp_feat].interpolate()
                a_data[interp_feat] = a_data[interp_feat].fillna(method='bfill')   
                a_data[interp_feat] = a_data[interp_feat].fillna(method='ffill') 
                new_data.append(a_data)
        data = pd.concat(new_data, axis=0) 
        del new_data    
    
    ################################
    # LGB Prediction：Patv
    ################################
    print('- LGB predict: Patv')
    normal_data = data.loc[~data.abnormal_patv, :]
    abnormal_data = data.loc[data.abnormal_patv, :]
    use_cols=['TurbID','Wspd','Wdir','Etmp','Itmp','Ndir','Prtv',]
    target_col = ['Patv']
    # Load pre-trained model
    lgb_model = lgb.Booster(model_file=settings['ab_lgb_model_path'])
    # replace abnormal values with predicted values
    abnormal_data.loc[:, target_col] = lgb_model.predict(abnormal_data[use_cols], num_iteration=lgb_model.best_iteration)
    data = pd.concat([normal_data, abnormal_data], axis=0)
    data = data.sort_values(by=['TurbID','Day','Tmstamp'],ascending=True)

    ################################
    # treat Prtv<0 and Patv<0, as 0
    ################################
    prtv_ab_mask = (data['Prtv'] < 0)
    data['Prtv'] = ~prtv_ab_mask * data['Prtv']
    patv_ab_mask = (data['Patv'] < 0)
    data['Patv'] = ~patv_ab_mask * data['Patv']

    ################################
    # remove the feature with 'abnormal_' prefix
    ################################
    ab_cols = [f for f in list(data) if 'abnormal_' in f]
    data = data.drop(ab_cols, axis=1)
    return data


def expand_basis_feats(data: pd.DataFrame, settings: dict):
    """Expand Basis Features"""
    # Envelope Residual
    max_paras = pd.read_csv(settings['max_paras_path'])
    data = pd.merge(data, max_paras[['TurbID','a','beta','b','mx']], how='left', left_on=['TurbID'], right_on=['TurbID'])
    data['max_patv1'] = data['a'] * data['Wspd']**data['beta'] + data['b']*data['Wspd']
    data['max_patv'] = np.min(data[['max_patv1','mx']].values, axis=1)
    data['patv_residual'] = data['max_patv'] - data['Patv']
    data.drop(['a','beta','b','mx','max_patv1'], axis=1, inplace=True)

    # Sine and Cosine Adjusted Wind Direction
    angle_diff_file = settings["diff_angle_path"]
    dfa = pd.read_csv(angle_diff_file)
    data = data.merge(dfa, how='left')
    data['w_angle'] = data['Ndir'] + data['Wdir']
    data['w_angle_adj'] = data['w_angle'] + data['diff_angle']
    data['w_angle_sin'] = np.sin(data['w_angle_adj'] * np.pi/180)
    data['w_angle_cos'] = np.cos(data['w_angle_adj'] * np.pi/180)
    print('Basis Features:', list(data))
    return data



def generate_tsfresh_feats(all_data: pd.DataFrame, settings: dict):
    """Use tsfresh Auto-Generate Features"""
    print('Feature Engineering: tsfresh auto generating features.')
    # Only for Patv, to expand more features
    data = all_data[['TurbID','time','Patv']]
    data = data.fillna(0)

    # Load feature settings
    saved_kind_to_fc_parameters = np.load(settings['tsfresh_params_path'], allow_pickle='TRUE').item()

    # Sample setting
    time_list = data['time'].unique()
    min_time = 144*3 # window_size
    sample_freq = 1 # sample stride
    sample_idxs = [i for i in np.arange(0,len(time_list), sample_freq) if i >= min_time-1]
    sample_times = time_list[sample_idxs]
    print('Total Sample Number: ', len(sample_times))

    trn_data_x = []
    if settings['is_submit'] == True:
        sample_times = [sample_times[-1]]
    for end_t in tqdm(sample_times):
        # get sample data
        sample_data = data[(data['time']>end_t-min_time) & (data['time']<=end_t)]
        # prepare x
        cur_time = max(sample_data.time)
        sample_data_x = sample_data[['TurbID','time','Patv']]
        # extract features
        extracted_features = extract_features(sample_data_x, column_id="TurbID", column_sort="time", 
                                              disable_progressbar=True,
                                              kind_to_fc_parameters=saved_kind_to_fc_parameters, n_jobs=5)
        extracted_features = extracted_features.reset_index().rename(columns={'index':'TurbID'})
        extracted_features['time'] = cur_time
        # collect features
        trn_data_x.append(extracted_features)

    # concat features
    trn_data_x = pd.concat(trn_data_x, axis=0).reset_index(drop=True)
    trn_data_x = trn_data_x.reset_index(drop=True)
    return trn_data_x


def skew(x):
    return x.skew()

def kurt(x):
    return x.kurt()

def cal_pert(x):
    return x.sum()/len(x)

def window_agg_fe(data: pd.DataFrame, 
                    time_fname: str='date',
                    tsid_fname: str='TurbID',
                    window_sizes: List[int]=[288],
                    feats_reals: List[str]=['Patv'],
                    feat_agg_dict: Dict=dict.fromkeys(['Patv'], [np.mean, np.max, np.min, np.var, np.median, kurt, skew]),
                    window_weight: str=None                  
                    ):
    """Window Statistical Features"""
    new_all_df = []
    for window_size in window_sizes:
        print('Feature Engineering: Aggregate a window of features (size=%d).' % window_size)
        a_df_list = []
        id_df_list = []
        # forloop each turbine
        for _, a_data in data.groupby(tsid_fname):
            a_df = a_data.sort_values(time_fname, ascending=True).copy(deep=True)
            a_df = a_df.reset_index(drop=True)
            id_df_list.append(a_df[[time_fname, tsid_fname]])
            agg_df_list = []
            # we aggregated the mean, maximum, minimum, variance, and median values within the history window.
            for f in feat_agg_dict.keys():
                agg_df = a_df[f].rolling(window_size, window_weight).agg(feat_agg_dict[f])
                agg_df.columns=[f'win{window_size}_{str(postfix)}_{f}' for postfix in list(agg_df)]
                agg_df_list.append(agg_df)
            agg_all_df = pd.concat(agg_df_list, axis=1)
            a_df_list.append(agg_all_df)    
        window_all_df = pd.concat(a_df_list, axis=0) # concat agg. features under the current window
        del agg_df_list, a_df_list
        new_all_df.append(window_all_df)
        del window_all_df
        gc.collect()
    new_all_df = pd.concat(new_all_df, axis=1) # concat agg. features under the different window
    id_all_df = pd.concat(id_df_list, axis=0)
    del id_df_list
    new_all_df = pd.concat([id_all_df, new_all_df], axis=1)
    return new_all_df



def decompose_fe(data: pd.DataFrame, 
                    time_fname: str='date',
                    tsid_fname: str='TurbID',
                    window_size: int=288,
                    cols: List[str]=['Patv']):
        """Series Decomposition Item"""
        print('Feature Engineering: Extract Seasonal Series.')
        all_df_list = []
        # forloop each turbine
        for _, a_data in data.groupby(tsid_fname):
            a_df = a_data.sort_values(time_fname, ascending=True).copy(deep=True)
            a_df = a_df.reset_index(drop=True)
            a_df_list = []
            for f in cols:
                a_df['seasonal_'+f] = a_df[f] - a_df[f].rolling(window_size).agg(np.mean)
                a_df_list.append(a_df[[time_fname, tsid_fname, 'seasonal_'+f]])
            a_df = pd.concat(a_df_list, axis=1)
            all_df_list.append(a_df)
        all_df = pd.concat(all_df_list, axis=0)
        return all_df



def generate_label(data: pd.DataFrame):
    """Label Generating: the t0+288th Patv"""
    print('Label Generating.')
    ts_IDs = data['TurbID'].unique()
    new_data = []
    for ts_ID in ts_IDs:
        a_data = data[data['TurbID']==ts_ID]
        a_data = a_data.sort_values(['TurbID', 'date'])
        a_data['target'] = a_data['Patv'].shift(-288)
        a_data['abnormal'] = a_data['abnormal'].shift(-288) # follow label to shift, for evaluation
        new_data.append(a_data)
    new_data = pd.concat(new_data, axis=0)
    return new_data



def timeseries_split(data: pd.DataFrame,
                        val_size: int=16,
                        test_size: int=15,
                        is_submit: bool=True):
    """dataset spliting: valid_size=15 day, test_size=15 day
    """
    print('Time Series Splitting.')
    if is_submit == False:
        date_range = data['Day'].sort_values().unique()
        val_cutoff_day = date_range[-(val_size+test_size+2)] # 留2天给预测集
        test_cutoff_day = date_range[-test_size-2]
        data['data_type'] = 'train'
        valid_idxs = data[ (data['Day'] >= val_cutoff_day) & (data['Day'] < test_cutoff_day)].index.to_list()
        test_idxs = data[ (data['Day'] >= test_cutoff_day) ].index.to_list()
        pred_idxs = data[data['target'].isnull()].index.to_list()
        data['data_type'].iloc[valid_idxs] = 'valid'
        data['data_type'].iloc[test_idxs] = 'test'
        data['data_type'].iloc[pred_idxs] = 'pred'  
    else:
        data['data_type'] = 'train'
        pred_idxs = data[data['target'].isnull()].index.to_list()
        data['data_type'].iloc[pred_idxs] = 'pred'
    print("After Data Splitting: train has %d, valid has %d, test has %d, pred has %d" % (len(data[data['data_type']=='train']),
                                                                            len(data[data['data_type']=='valid']),
                                                                            len(data[data['data_type']=='test']),
                                                                            len(data[data['data_type']=='pred'])))
    return data
    


def standardlize(data: pd.DataFrame,
                 real_feats: List[str],
                 scaler_save_path: str,
                 is_submit: bool):
    """Standardlizing"""
    print('Standardlizing.')
    ts_IDs = data['TurbID'].unique()
    if is_submit == False:
        new_data = []
        # save mu and std of each continuous feature of each turbine 
        scaler_dict = {'real_feats':real_feats}
        for ts_ID in ts_IDs:
            a_data = data[data['TurbID']==ts_ID]
            trn_data = a_data[a_data['data_type']=='train'][real_feats]
            trn_mean = trn_data.mean().to_list()
            trn_std = trn_data.std().to_list()
            scaler = (trn_mean, trn_std)
            a_data[real_feats] = (a_data[real_feats] - trn_mean) / trn_std
            new_data.append(a_data)
            scaler_dict[ts_ID] = scaler
        new_data = pd.concat(new_data, axis=0).reset_index(drop=True)   
        np.save(scaler_save_path, scaler_dict) # save mu and std
    # if online, use saved mu and std to standardlize date
    else:
        scaler_dict = np.load(scaler_save_path, allow_pickle='True').item()
        real_feats = scaler_dict['real_feats']
        new_data = []
        for ts_ID in ts_IDs:
            a_data = data[data['TurbID']==ts_ID]
            trn_mean = scaler_dict[ts_ID][0]
            trn_std = scaler_dict[ts_ID][1]
            a_data[real_feats] = (a_data[real_feats] - trn_mean) / trn_std
            new_data.append(a_data)
        new_data = pd.concat(new_data, axis=0).reset_index(drop=True)
    return new_data 



def inverse_standardize(data: pd.DataFrame,
                        scaler_save_path: str,
                        is_submit: bool):
    """Inverse standardlization"""
    scaler_dict = np.load(scaler_save_path, allow_pickle='True').item()
    real_feats = scaler_dict['real_feats']
    real_feats = [re.sub('[^A-Za-z0-9_]+', '', x) for x in real_feats]
    target_idx = real_feats.index('target')
    new_data = []
    ts_IDs = data['TurbID'].unique()
    for ts_ID in ts_IDs:
        a_data = data[data['TurbID']==ts_ID]
        if is_submit == False:
            a_data[real_feats] = a_data[real_feats] * scaler_dict[ts_ID][1] + scaler_dict[ts_ID][0]
        a_data['pred_target'] = a_data['pred_target'] * scaler_dict[ts_ID][1][target_idx] + scaler_dict[ts_ID][0][target_idx]
        new_data.append(a_data)
    new_data = pd.concat(new_data, axis=0).reset_index(drop=True)    
    return new_data



def data_preprocess(settings):
    # ================
    # Parameter and Data Loading
    # ================
    all_df = load_dataset(settings)

    # ================
    # Abnormal Preprocessing
    # ================
    all_df = abnormal_preprocess(all_df, settings)

    # ================
    # Basis Feature Expanding
    # ================    
    all_df = expand_basis_feats(all_df, settings)

    # add time col
    all_df = all_df.sort_values(['TurbID','Day','Tmstamp'], ascending=True)
    day_num = len(all_df['Day'].unique())
    turb_num = len(all_df['TurbID'].unique())
    time_col = [j for i in range(turb_num) for j in range(int(len(all_df)/turb_num))]
    all_df['time'] = time_col   

    # tsfresh auto-generate features
    tsfresh_all_df = generate_tsfresh_feats(all_df, settings)

    # ================
    # Feature Engineerging
    # ================
    feats_reals=['Wspd','Etmp','Itmp','Prtv','Patv','max_patv','w_angle_sin','w_angle_cos']
    wa_df = window_agg_fe(all_df,
                            window_sizes=[288],
                            window_weight=None,
                            feats_reals=feats_reals,
                            feat_agg_dict=dict.fromkeys(feats_reals, [np.mean, np.max, np.min, np.var, np.median]))
    dc_df = decompose_fe(all_df, window_size=288, cols=['Patv'])
    new_feat_df = pd.merge(wa_df, dc_df, how='left', left_on=['date','TurbID'], right_on=['date','TurbID'])
    all_df = pd.merge(all_df, new_feat_df, how='left', left_on=['date','TurbID'], right_on=['date','TurbID'])
    del new_feat_df

    # concat tsfresh features and manually generated feature
    all_df = pd.merge(all_df, tsfresh_all_df, how='left', left_on=['time','TurbID'], right_on=['time','TurbID'])

    # ================
    # Label Generating
    # ================
    all_df = generate_label(all_df)
    # remove the empty rows caused by feature enginneering
    all_df = all_df[all_df['time']>=144*3-1]
    all_df = all_df.reset_index(drop=True)
    print('Current Feats:', list(all_df))

    # ================
    # Data Spliting
    # ================
    all_df = timeseries_split(all_df, val_size=15, test_size=15, is_submit=settings['is_submit'])

    # ================
    # Standardlizaiton
    # ================
    cat_feats = ['TurbID', 'abnormal', 'Day', 'Tmstamp', 'date', 'data_type']
    real_feats = [f for f in list(all_df) if f not in cat_feats]
    all_df = standardlize(all_df, real_feats, settings["scaler_save_path"], settings['is_submit'])


    # ================
    # Save preprocessed dataset
    # ================
    if settings['is_submit'] == False:
        print('Saving preprocessed data to ', settings['preprocess_data_path'])
        all_df.to_csv(settings['preprocess_data_path'], index=False)
    return all_df


if __name__=="__main__":
    settings = prep_env()
    all_df = data_preprocess(settings)