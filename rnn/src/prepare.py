# -*-Encoding: utf-8 -*-
################################################################################
"""
Description: Prepare the experimental settings
"""
import os
from rnn.src.hierarchy.hierarchical_dicts import *
path = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(path)!='src':
    path = os.path.abspath(os.path.join(path, '..'))
envelope_path = os.path.abspath(os.path.join(path, 'stored_process'))

def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    args = {
        # required args
        #"data_file": "/home/notebook/data/group/intention_rec/TimeSeries/KDD2022/wtbdata_245days.csv",            
        "model_file_name": 'RNN_multiple_2022_07_19_07',   # model saved in working_dir/checkpoints/model_file_name
        'model_dir': 'paddlepaddle',
        'paddlepaddle_model': 'RNN_multiple', 
        "ensemble_preps": [[(0, 12), ['prep_HH_2_HD1_128_HD2_16_OUT_12_PD_140_AT_True', 'prep_HH_2_HD1_128_HD2_16_OUT_12_PD_None_AT_False', 'prep_HH_2_HD1_128_HD2_16_OUT_12_PD_None_AT_True', 'prep_HH_2_HD1_128_HD2_8_OUT_12_PD_140_AT_False', 'prep_HH_2_HD1_128_HD2_8_OUT_12_PD_140_AT_True', 'prep_HH_2_HD1_128_HD2_8_OUT_12_PD_None_AT_False', 'prep_HH_2_HD1_128_HD2_8_OUT_12_PD_None_AT_True', 'prep_HH_2_HD1_64_HD2_16_OUT_12_PD_140_AT_False', 'prep_HH_2_HD1_64_HD2_16_OUT_12_PD_140_AT_True', 'prep_HH_2_HD1_64_HD2_16_OUT_12_PD_None_AT_False', 'prep_HH_2_HD1_64_HD2_8_OUT_12_PD_140_AT_False', 'prep_HH_2_HD1_64_HD2_8_OUT_12_PD_None_AT_False', 'prep_HH_4_HD1_128_HD2_16_OUT_12_PD_None_AT_True', 'prep_HH_4_HD1_128_HD2_8_OUT_12_PD_140_AT_False', 'prep_HH_4_HD1_128_HD2_8_OUT_12_PD_None_AT_True', 'prep_HH_4_HD1_64_HD2_16_OUT_12_PD_140_AT_False', 'prep_HH_4_HD1_64_HD2_8_OUT_12_PD_140_AT_True', 'prep_HH_4_HD1_64_HD2_8_OUT_12_PD_None_AT_True', 'prep_HH_7_HD1_128_HD2_16_OUT_12_PD_None_AT_True', 'prep_HH_7_HD1_64_HD2_8_OUT_12_PD_140_AT_True']], [(12, 36), ['prep_HH_2_HD1_128_HD2_16_OUT_36_PD_140_AT_False', 'prep_HH_2_HD1_128_HD2_16_OUT_36_PD_None_AT_False', 'prep_HH_2_HD1_128_HD2_16_OUT_36_PD_None_AT_True', 'prep_HH_2_HD1_128_HD2_8_OUT_36_PD_140_AT_False', 'prep_HH_2_HD1_128_HD2_8_OUT_36_PD_140_AT_True', 'prep_HH_2_HD1_128_HD2_8_OUT_36_PD_None_AT_False', 'prep_HH_2_HD1_128_HD2_8_OUT_36_PD_None_AT_True', 'prep_HH_2_HD1_64_HD2_16_OUT_36_PD_140_AT_True', 'prep_HH_2_HD1_64_HD2_16_OUT_36_PD_None_AT_False', 'prep_HH_2_HD1_64_HD2_16_OUT_36_PD_None_AT_True', 'prep_HH_2_HD1_64_HD2_8_OUT_36_PD_140_AT_True', 'prep_HH_2_HD1_64_HD2_8_OUT_36_PD_None_AT_True', 'prep_HH_4_HD1_128_HD2_16_OUT_36_PD_140_AT_True', 'prep_HH_4_HD1_128_HD2_16_OUT_36_PD_None_AT_True', 'prep_HH_4_HD1_128_HD2_8_OUT_36_PD_140_AT_True', 'prep_HH_4_HD1_128_HD2_8_OUT_36_PD_None_AT_True', 'prep_HH_4_HD1_64_HD2_16_OUT_36_PD_None_AT_True', 'prep_HH_7_HD1_128_HD2_16_OUT_36_PD_140_AT_True', 'prep_HH_7_HD1_64_HD2_16_OUT_36_PD_140_AT_False', 'prep_HH_7_HD1_64_HD2_8_OUT_36_PD_140_AT_False']], [(36, 72), ['prep_HH_2_HD1_128_HD2_16_OUT_72_PD_140_AT_False', 'prep_HH_2_HD1_128_HD2_16_OUT_72_PD_140_AT_True', 'prep_HH_2_HD1_128_HD2_16_OUT_72_PD_None_AT_True', 'prep_HH_2_HD1_128_HD2_8_OUT_72_PD_140_AT_True', 'prep_HH_2_HD1_128_HD2_8_OUT_72_PD_None_AT_True', 'prep_HH_2_HD1_64_HD2_16_OUT_72_PD_140_AT_False', 'prep_HH_2_HD1_64_HD2_16_OUT_72_PD_None_AT_False', 'prep_HH_2_HD1_64_HD2_16_OUT_72_PD_None_AT_True', 'prep_HH_4_HD1_128_HD2_16_OUT_72_PD_140_AT_True', 'prep_HH_4_HD1_128_HD2_16_OUT_72_PD_None_AT_True', 'prep_HH_4_HD1_128_HD2_8_OUT_72_PD_140_AT_True', 'prep_HH_4_HD1_64_HD2_16_OUT_72_PD_140_AT_False', 'prep_HH_4_HD1_64_HD2_16_OUT_72_PD_140_AT_True', 'prep_HH_4_HD1_64_HD2_16_OUT_72_PD_None_AT_False', 'prep_HH_4_HD1_64_HD2_16_OUT_72_PD_None_AT_True', 'prep_HH_4_HD1_64_HD2_8_OUT_72_PD_140_AT_True', 'prep_HH_4_HD1_64_HD2_8_OUT_72_PD_None_AT_False', 'prep_HH_4_HD1_64_HD2_8_OUT_72_PD_None_AT_True', 'prep_HH_7_HD1_128_HD2_16_OUT_72_PD_140_AT_True', 'prep_HH_7_HD1_64_HD2_16_OUT_72_PD_None_AT_True']], [(72, 144), ['prep_HH_2_HD1_128_HD2_16_OUT_144_PD_140_AT_False', 'prep_HH_2_HD1_128_HD2_16_OUT_144_PD_None_AT_False', 'prep_HH_2_HD1_128_HD2_8_OUT_144_PD_140_AT_True', 'prep_HH_2_HD1_128_HD2_8_OUT_144_PD_None_AT_False', 'prep_HH_2_HD1_128_HD2_8_OUT_144_PD_None_AT_True', 'prep_HH_2_HD1_64_HD2_16_OUT_144_PD_None_AT_False', 'prep_HH_2_HD1_64_HD2_16_OUT_144_PD_None_AT_True', 'prep_HH_2_HD1_64_HD2_8_OUT_144_PD_140_AT_False', 'prep_HH_2_HD1_64_HD2_8_OUT_144_PD_None_AT_False', 'prep_HH_4_HD1_128_HD2_8_OUT_144_PD_140_AT_True', 'prep_HH_4_HD1_64_HD2_16_OUT_144_PD_140_AT_True', 'prep_HH_4_HD1_64_HD2_8_OUT_144_PD_None_AT_True', 'prep_HH_7_HD1_128_HD2_16_OUT_144_PD_140_AT_True', 'prep_HH_7_HD1_128_HD2_16_OUT_144_PD_None_AT_True', 'prep_HH_7_HD1_128_HD2_8_OUT_144_PD_140_AT_True', 'prep_HH_7_HD1_128_HD2_8_OUT_144_PD_None_AT_True', 'prep_HH_7_HD1_64_HD2_16_OUT_144_PD_140_AT_False', 'prep_HH_7_HD1_64_HD2_16_OUT_144_PD_140_AT_True', 'prep_HH_7_HD1_64_HD2_16_OUT_144_PD_None_AT_True', 'prep_HH_7_HD1_64_HD2_8_OUT_144_PD_140_AT_True']], [(144, 288), ['prep_HH_2_HD1_128_HD2_16_OUT_288_PD_140_AT_False', 'prep_HH_2_HD1_128_HD2_16_OUT_288_PD_140_AT_True', 'prep_HH_2_HD1_128_HD2_16_OUT_288_PD_None_AT_False', 'prep_HH_2_HD1_128_HD2_16_OUT_288_PD_None_AT_True', 'prep_HH_2_HD1_128_HD2_8_OUT_288_PD_140_AT_False', 'prep_HH_2_HD1_128_HD2_8_OUT_288_PD_140_AT_True', 'prep_HH_2_HD1_128_HD2_8_OUT_288_PD_None_AT_True', 'prep_HH_2_HD1_64_HD2_16_OUT_288_PD_140_AT_False', 'prep_HH_2_HD1_64_HD2_16_OUT_288_PD_140_AT_True', 'prep_HH_2_HD1_64_HD2_16_OUT_288_PD_None_AT_False', 'prep_HH_2_HD1_64_HD2_8_OUT_288_PD_None_AT_True', 'prep_HH_4_HD1_128_HD2_16_OUT_288_PD_None_AT_False', 'prep_HH_4_HD1_128_HD2_8_OUT_288_PD_None_AT_False', 'prep_HH_4_HD1_64_HD2_16_OUT_288_PD_140_AT_True', 'prep_HH_4_HD1_64_HD2_16_OUT_288_PD_None_AT_True', 'prep_HH_4_HD1_64_HD2_8_OUT_288_PD_None_AT_False', 'prep_HH_7_HD1_128_HD2_16_OUT_288_PD_None_AT_True', 'prep_HH_7_HD1_128_HD2_8_OUT_288_PD_140_AT_True', 'prep_HH_7_HD1_64_HD2_8_OUT_288_PD_140_AT_False', 'prep_HH_7_HD1_64_HD2_8_OUT_288_PD_140_AT_True']]],
       'final_adjust': True,
        
        'input_len': 288*7,
        'hid_sz': 64,
        'hid2_sz': 32,
        "mode": 'train',  # 
        'loss_w': 0.8,
        'loss2_w': 0.2,
        'use_gpu': True,
        #"data_file": "/home/notebook/data/group/intention_rec/TimeSeries/KDD2022/wtbdata_10days.csv",   
        "data_file": "/home/notebook/data/group/intention_rec/TimeSeries/KDD2022/wtbdata_245days.csv",   
        "location_file": "/home/notebook/data/group/intention_rec/TimeSeries/KDD2022/sdwpf_baidukddcup2022_turb_location.CSV",        # 
        "path_to_test_x": "/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/PaddleSpatial/apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/data/sdwpf_baidukddcup2022_test_toy/test_x/0001in.csv",
        "path_to_test_y": "/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/PaddleSpatial/apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/data/sdwpf_baidukddcup2022_test_toy/test_y/0001out.csv",
        "angle_diff_file": os.path.join(path, "stored_process/diff_angle.csv"),
        
        ####
        
        'test_days': 15,  # Number of days for validation
        'val_days': 15,   # Number of days for testing
        "output_len": 288,
        'partial_data': 171, #None, #171,
        'output_attention': True,
        ## fintune
        'finetune': True,
        'finetune_epochs': 2,
        
        
        # dataloader
        
        'shuffle': True,
        'num_workers': 1, # num of workers for loading data, not training or evaluation
        'drop_last': False,
        'use_shared_memory': False,
        'preprocess_method': 'multi_var',
        
        ### scaler
        'divide_by_max_patv': True, # 
        'max_patv': 1521.6,  # ,
        'scale_method': 'multi_scaler', # 
        'adaptive_norm': False,
        'lstm_dec_lookbk': 6, # lstm dec input is the mean of history start from lstm_dec_lookbk
        
        'more_median_cols': ['w_angle_sin', 'w_angle_cos'], # more median cols to calcluate
        'enc_f_cols': ['etmp_median'],
        'f_cols': [],
        
        'value2_cols': ['ws_median'],
        'enc_f2_cols': ['w_angle_sin_median', 'w_angle_cos_median', 'mask_mean'],
        'f2_cols': ['sin_tm', 'cos_tm'],
        'normal_noise': False,
        ### optional args
        'hierarchical': True, 
        'hierarchical_mask': True,
        'hie_dic': hierarchy_dict2, # required if hierarchical is not None
        "base_nodes": list(range(1,135)),   # every turbine
        'generate_noisy_sample': True, 
        'shift_time_based_on_Etmp': False, # whether to shift 00:00 according to lowest temperature of the day
        'fill_unknown_method': 'interpolate',   # lgb, interpolate
        'envelope_abnormal': True,
        'envelope_file': os.path.join(envelope_path, 'max_paras.csv'),
        'abnormal_lgb_file': os.path.join(envelope_path, 'lgb_abnormal_full.txt'),
        'lgb_abnormal': True,
        # model
        "checkpoints": "checkpoints",
        'masked_loss': True,
        'valid_type': 1, # 0, kdd_score, 1 loss
        "start_col": 3,
        "dropout": 0.05,
        "train_epochs": 20,
        "batch_size": 32,
        "patience": 30,
        "lr": 1e-5,
        "lr_adjust": "type1",
        "gpu": 0,
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
        'verbose': True,
        'debug': False,
        "stored_args": { # store updated args during calculation
        }
    }
    return args
