# -*-Encoding: utf-8 -*-
"""
Description: Parameter Setting
Authors: aihongfeng
Date:    2022/05/20
Function:

"""
import os

def prep_env():
    path = os.path.dirname(os.path.abspath(__file__))
    exp_ver = 16
    settings = {
        # when online infer, platform will change the following test paths
        "path_to_test_x": "/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/PaddleSpatial/apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/data/test_x/0001in.csv",
        "path_to_test_y": "/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/PaddleSpatial/apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/data/test_y/0001out.csv",
        "data_path": "/home/notebook/data/group/intention_rec/TimeSeries/KDD2022",
        "filename": "wtbdata_245days.csv",
        
        # data preprocess
        "diff_angle_path":os.path.join(path, "diff_angle.csv"),                                 # for wind dir adjustment
        "ab_lgb_model_path":os.path.join(path, "lgb_abnormal_fix_Patv_regressor.txt"),          # for abnormal Patv
        "max_paras_path":os.path.join(path, "max_paras.csv"),                                   # for envelope feature
        "preprocess_data_path": os.path.join(path, f"wtbdata_245days_preprocess{exp_ver}.csv"), # the save path of the preprocessed dataset
        "scaler_save_path": os.path.join(path, f"scaler{exp_ver}.npy"),                         # mu and std for standardization
        "tsfresh_params_path": os.path.join(path, f"kind_to_fc_parameters_top59.npy"),          # tsfresh feature settings
        
        # model training
        "lgb_model_save_path": os.path.join(path, f"tmodels/lgb_regressor{exp_ver}.txt"),               # the save path of model
        "lgb_feat_import_save_path":os.path.join(path, f"tmodels/lgb_feat_import{exp_ver}.png"),        # the save path of feature importance figure
        
        # others
        "checkpoints": "checkpoints",
        "start_col": 3,
        "total_size": (432+288)*134, # total_size=(window_size+pred_len)*134
        "pred_file": "predict.py",
        "framework": "base",
        "is_submit":  True, # whether go online inference or offline training
    }
    print("The experimental settings are: \n{}\n".format(str(settings)))
    return settings