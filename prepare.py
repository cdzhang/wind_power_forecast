# -*-Encoding: utf-8 -*-
"""
Description: Prepare the experimental settings
"""

def prep_env():
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """

    args = {
        "checkpoints": "checkpoints",
        "data_file": "/home/notebook/data/group/intention_rec/TimeSeries/KDD2022/wtbdata_245days.csv",   
        #"data_file": "/home/notebook/data/group/intention_rec/TimeSeries/KDD2022/wtbdata_10days.csv",   
        "path_to_test_x": "/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/PaddleSpatial/apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/data/test_x/0001in.csv",
        "path_to_test_y": "/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/PaddleSpatial/apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/data/test_y/0001out.csv",
        'final_adjust': True,
        "start_col": 3,
        'model_dirs': ['rnn', 'lgb_model'],  # model dirs for ensemble
        "pred_file": "predict.py",
        'framework': 'paddlepaddle'
    }
    return args
