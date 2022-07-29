import pandas as pd
import numpy as np
import sys
import os
import gc
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print(path, 'added')

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.abspath(os.path.join(path, '..'))
add_path(path)

from common import *
from rnn.src.prepare import prep_env

global_args = prep_env()
global_args = load_args(global_args)

global_args['mode'] = 'submit'
module = __import__(global_args['model_dir'])


def adjust_prediction(pred, df_enc, args, fd=288, bd=6):
    patv = pd.pivot_table(df_enc, index=['Day','Tmstamp'], columns='TurbID', values='Patv')[args['base_nodes']].values
    last = np.nanmedian(patv[-bd:,:], axis=0).reshape(134, 1, 1)
    last = np.repeat(last, fd, axis=1)
    
    t = np.arange(fd)
    weights = (t-fd)**2/fd**2
    weights = weights.reshape(1, fd, 1)
    weights = np.repeat(weights, 134, axis=0)
    gap = last - pred[:,:fd,:]
    gap[pd.isna(gap)] = 0
    pred_a = pred.copy()
    pred_a[:,:fd,:] = pred[:,:fd, :] +gap*weights
    return pred_a

class EnsembleWrapper:
    def __init__(self,  rg, preps):
        self.output_len = rg[1]
        self.start, self.end = rg
        self.preps = preps 
        self.args_list = [load_args(__import__(prep).prep_env()) for prep in preps]
        for args in self.args_list:
            args['mode'] = 'submit'
        
        self.model_wrappers = [module.load_trained_model_wrapper(args) for args in self.args_list]
        
        
    def predict(self, load_data_returns, load_data_returns_tmp):
        
        
        
        preds = []
        for args, model_wrapper in zip(self.args_list, self.model_wrappers):
            if args['model_file_name'].find('ETMP')>0:
                (values, mask, value_cols, enc_f_cols, f_cols, value2_cols, enc_f2_cols, f2_cols, extend_features) = load_data_returns_tmp
            else:
                (values, mask, value_cols, enc_f_cols, f_cols, value2_cols, enc_f2_cols, f2_cols, extend_features) = load_data_returns
            extend_features = extend_features.iloc[:self.output_len]
            x_input = values.iloc[-args['input_len']:]
            x_mask = mask.iloc[-args['input_len']:]  
            scaler = args['stored_args']['stored_scaler']
            scaler2 = args['stored_args']['stored_scaler2']
            dataset = module.data_lstm_enc_f.MyDataset(x_input, x_mask, 
                    value_cols, enc_f_cols, f_cols, value2_cols, enc_f2_cols, f2_cols, args, scaler, scaler2, extend_features)
            dataloader = module.data_lstm_enc_f.create_data_loader([dataset], args)[0]
            x_enc, x2_enc, extend_feature, mask_enc = list(dataloader)[0]
            if len(args['f_cols'])>0:
                f_dec = extend_feature
            else:
                f_dec = None
            if len(args['f2_cols']) > 0:
                f2_dec = extend_feature
            else:
                f2_dec = None
            
            if args['paddlepaddle_model'] == 'LSTM_enc2_tp2':
                scaler = args['stored_args']['stored_scaler']
                u, std = scaler.mean, scaler.std
                y, _ = model_wrapper.model(u, std, x_enc, x2_enc, f_dec=f_dec, f2_dec=f2_dec)
            else:
                y, _ = model_wrapper.model(x_enc, x2_enc, f_dec=f_dec, f2_dec=f2_dec)
            pred = scaler.inverse_transform(y)*args['max_patv']
            pred = pred[:,:,:134].clip(0, args['max_patv'])
            preds.append(pred)
        #preds = np.concatenate(preds, axis=0)
        #preds = np.median(preds, axis=0, keepdims=True)
        pred_median = np.concatenate(preds, axis=0)
        pred_median = np.median(pred_median, axis=0, keepdims=True)
        return preds, pred_median[:,self.start:self.end, :]

    
if global_args['paddlepaddle_model'] == 'RNN_multiple':
    ews = [EnsembleWrapper(*x) for x in global_args['ensemble_preps']]
    tmp_args = __import__('prepare_tmp').prep_env()
    tmp_args['mode'] = 'submit'
else:
    wrapper = module.load_trained_model_wrapper(global_args)

    
def forecast(settings, return_single_preds=False):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions
    """
    df_enc = pd.read_csv(settings['path_to_test_x'])
    
    if global_args['paddlepaddle_model'] == 'RNN_multiple':
        load_data_returns = module.data_lstm_enc_f.load_data(global_args, df_enc, True)
        load_data_returns_tmp = module.data_lstm_enc_f.load_data(tmp_args, df_enc, True)
        pred_medians = []
        preds_list = []
        for ew in ews:
            preds, pred_median = ew.predict(load_data_returns, load_data_returns_tmp)
            if return_single_preds:
                preds_list.append(preds)
            pred_medians.append(pred_median)
            
        if return_single_preds:
            return preds_list, pred_medians
        pred = np.concatenate(pred_medians, axis=1)
        pred = np.einsum('ijk->kji', pred)
        if settings['final_adjust']:
            return adjust_prediction(pred, df_enc, global_args, fd=60, bd=10)
        else:
            return pred
    pred = wrapper.predict(df_enc)
    if pred.std(axis=1).mean() <= 10 or pred.std(axis=0).mean() <= 10:
        pred += np.random.randn(134, 288, 1) * 0.015
    gc.collect()
    return pred







if __name__ == '__main__':
    print(forecast(args).shape)
