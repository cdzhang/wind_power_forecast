# Spatial Dynamic Wind power forecasting using lightGBM and multi-variate LSTM with hierarchical coherence constraints

This is the implementation of ensembling method of  `lightGBM` and multi-variate `LSTM` model for wind Spatial Dynamic Wind Power Forecasting, [KDD CUP 2022]([Baidu KDD CUP 2022 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction)) hold by Baidu.

`lightGBM` makes middle-term forecasts (24-48hours) and `LSTM` makes short-term forecasts (0-24hours). Multiple forecasts are made by both `lightGBM` and `LSTM` and ensemble forecasts are made separately, then the two forecasts are concatenated to form the final forecast.

## train

`lightGBM`model

in `lgb_model/src/prepare.py`, set `is_submit` to `False`.

```python
cd lgb_model/src  # go to lgb_model/src directory first
python wind_turbine_data.py
python predict.py
```

`LSTM` model

```python
cd rnn/src   # go to rnn/src directory first
python pipline.py prepare_example.py
```

When a model is trained successively, add it to the `rnn/src/prepare.py` by updating the key of `ensemble_preps` manually or just run `python cal_trained.py` , the script will search `rnn/src/checkpoints` for trained model and automatically generate `prepare.py` file in `rnn/src`. Correct path should be set in `cal_trained.py` first.

## Forecast

set `is_submit` in `lgb_model/src/prepare.py` to `True` first. 

Then model `predict.py` in `src_ensemble` directory is ready for forecasting tasks.

## Code Structure

```
── images   # images for readme.md
├── lgb_model
│   └── src
│       ├── prepare.py                            # parameter setting
│       ├── wind_turbine_data.py                  # data preparing.
│       ├── predict.py                            # call for the prediction of the given test dataset
│       ├── model_wrapper.py                      # model training
│       ├── metrics.py                            # KDD evaluation metrics for the offline evaluation.
│       ├── max_paras.csv                         # the parameters of the wind speed - max power curve of each turbine
│       ├── scaler16.npy                          # it contains mu and std of each feature of each turbine
│       ├── diff_angle.csv                        # the wind direction adjustment angle for each turbine
│       ├── kind_to_fc_parameters_top59.npy       # the tsfresh top59 target-relevant feature settings
│       ├── lgb_abnormal_fix_Patv_regressor.txt   # the LightGBM model to repair the abnormal Patv values
│       ├── tmodels                               # the pre-trained lightGBM models with different parameter settings
│       │   ├── regressor_gbdt_seed1000_ss08.txt
│       │   ├── regressor_gbdt_seed1000.txt
│       │   ├── ...
│       └── README.md
├── pack.sh    # script to pack code into zip
├── predict.py # predict API
├── prepare.py # env for total model 
├── readme.md 
└── rnn        # RNN model
    ├── __init__.py
    └── src
        ├── cal_trained.py   # search for rnn/src/checkpoints and automaticall generate rnn/src/prepare.py
        ├── checkpoints      # store trained models and updated envs
        │   ├── RNN_hh_2_hidsz1_128_hidsz2_16_out_12_pd_140_attn_True   # trained model
        │   ├── RNN_hh_2_hidsz1_128_hidsz2_16_out_12_pd_140_attn_Trueupdated_args  # updated envs
        |   ├── RNN* # stored trained model parameters and updated envs
        ├── common.py        # common tools for load models and args
        ├── data_preprocess.py  # data processing 
        ├── generate_preps_2.py # generate multiple env files (each for one model)
        ├── generate_preps.py   # generate multiple env files
        ├── hierarchy           # hierarchical structure of wind turbines 
        │   ├── hierarchical_dicts.py
        ├── __init__.py 
        ├── metrics.py          # metrics for model results
        ├── paddlepaddle        # paddlepaddle model
        │   ├── data_lstm_enc_f.py   # calculate data loader
        │   ├── __init__.py
        │   ├── model.py             # define model
        │   ├── model_wrapper.py     # API for train and predict
        │   └── tools.py             # common tools
        ├── pipeline.py              # train a model from command line
        ├── predict.py               # API for model prediction
        ├── prepare_ensemble_origin.py  # automatical generate rnn/src/prepare.py
        ├── prepare_example.py          # example env file for single model
        ├── prepare_origin2.py          # generate multiple prepare files
        ├── prepare_origin.py           # generate multiple prepare files
        ├── prepare.py                  # env files for rnn model
        ├── prepare_tmp.py              # generate multiple prepare files
        ├── prep_HH_2_HD1_128_HD2_16_OUT_12_PD_140_AT_False.py  # generated prepare file
        |── prep_HH*.py # generated prepare files
        └── stored_process  # store process of data preprocessing
            ├── diff_angle.csv   # wind turbine angle adjustment
            ├── lgb_abnormal_fix_Patv_regressor.txt # lgb fill unknown patv v1
            ├── lgb_abnormal_full.txt   # lgb fill unkonwn patv v4
            ├── lgb_abnormal_o.txt      # lgb fill unkonwn patv v3 
            ├── lgb_abnormal.txt        # lgb fill unkonwn patv v2
            ├── max_paras.csv           # max patv - wind power curve results
            ├── noisy_sample.pkl        # normal examples of patv partitioned by wind speed
```
