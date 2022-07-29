# -*-Encoding: utf-8 -*-



################################################################################
"""
Description: Prepare the experimental settings
"""
import os
from hierarchy.hierarchical_dicts import *
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
        #"data_file": "/home/notebook/data/group/intention_rec/TimeSeries/KDD2022/wtbdata_245days.csv",            # 历史数据文件
        "model_file_name": 'RNNETMP_hh_4_hidsz1_64_hidsz2_16_out_216_pd_140_attn_True',   # 模型保存在 working_dir/checkpoints/model_file_name
        'model_dir': 'paddlepaddle',
        'paddlepaddle_model': 'LSTM_enc_f2', 
        'input_len': 288*4,
        'hid_sz': 64,
        'hid2_sz': 16,
        "mode": 'train',  # ['train', 'val', 'test', 'submit']  train: 数据集划分为train, val, 训练模型。  val: 训练集划分为train,val,test。test: load model 在测试集上测试。submit: load model, 提交
        'loss_w': 0.8,
        'loss2_w': 0.2,
        'use_gpu': True,
        #"data_file": "/home/notebook/data/group/intention_rec/TimeSeries/KDD2022/wtbdata_10days.csv",   
        "data_file": "/home/notebook/data/group/intention_rec/TimeSeries/KDD2022/wtbdata_245days.csv",   
        "location_file": "/home/notebook/data/group/intention_rec/TimeSeries/KDD2022/sdwpf_baidukddcup2022_turb_location.CSV",        # 位置数据文件
        "path_to_test_x": "/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/PaddleSpatial/apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/data/sdwpf_baidukddcup2022_test_toy/test_x/0001in.csv",
        "path_to_test_y": "/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/PaddleSpatial/apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/data/sdwpf_baidukddcup2022_test_toy/test_y/0001out.csv",
        "angle_diff_file": os.path.join(path, "stored_process/diff_angle.csv"),
        ####
        'test_days': 15,  # Number of days for validation
        'val_days': 15,   # Number of days for testing
        "output_len": 216,
        'partial_data': 140, #None, #171,
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
        'divide_by_max_patv': True, # 是否除以max_patv
        'max_patv': 1521.6,  # patv最大值,
        'scale_method': 'multi_scaler', # 归一化方式, 自定义 [False, max_value], 然后自己写方法
        'adaptive_norm': False,
        'lstm_dec_lookbk': 6, # lstm 输入的回看长度
        
        'more_median_cols': ['w_angle_sin', 'w_angle_cos'], # 更多的median col
        #'enc_f_cols': ['etmp_median'],
        'enc_f_cols': [],
        'f_cols': [],
        
        'value2_cols': ['etmp_median', 'ws_median'],
        'enc_f2_cols': ['w_angle_sin_median', 'w_angle_cos_median', 'mask_mean'],
        'f2_cols': ['sin_tm', 'cos_tm'],
        'normal_noise': False,
        ### optional args
        'hierarchical': True, 
        'hierarchical_mask': True,
        'hie_dic': hierarchy_dict3, # required if hierarchical is not None
        "base_nodes": list(range(1,135)),   # 每个风机
        'generate_noisy_sample': True, 
        'envelope_abnormal': True,
        'envelope_file': os.path.join(envelope_path, 'max_paras.csv'),
        'abnormal_lgb_file': os.path.join(envelope_path, 'lgb_abnormal_full.txt'),
        'lgb_abnormal': True,
        # model
        "checkpoints": "checkpoints",
        'masked_loss': True,
        'valid_type': 1, # 0, 用kdd_score, 1用 loss
        "start_col": 3,
        "dropout": 0.05,
        "train_epochs": 20,
        "batch_size": 32,
        "patience": 20,
        "lr": 1e-4,
        "lr_adjust": "type1",
        "gpu": 0,
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
        'verbose': True,
        'debug': False,
        "stored_args": {  
            #'stored_scaler'
            
        }
    }
    return args
