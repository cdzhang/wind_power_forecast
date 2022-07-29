import pandas as pd
import numpy as np
import os
from rnn.src.data_preprocess import *
from paddle.io import Dataset, DataLoader
import paddle

def load_data(args, df=None, submit_return=False):
    """
    args: return of prep_env()
    df: raw data, if None, read from args['data_file']
    mode: train, val, test, submit
    submit_return: bool, used for ensemble submission
    return: dataloaders, datasets
    """
    
    processor = DataPreprocessor(args)
    values, mask, value_cols, extend_features = processor.process_data_multi_variate(df)  # data preprocessing
    
    values[value_cols] /= args['max_patv']   # divide patv of turbines and turbine groups by max_patv first
    total_size = len(values)
    
    enc_f_cols = args['enc_f_cols']          # encoder features of lstm layer 1
    f_cols = args['f_cols']                  # features of layer 1
    value2_cols = args['value2_cols']        # values of layer 2
    enc_f2_cols = args['enc_f2_cols']        # encoder features of layer 2
    f2_cols = args['f2_cols']                # features of layer 2
    
    stored_args = args['stored_args']
    stored_args['v_sz'] = len(value_cols)       # value size
    stored_args['enc_f_sz'] = len(enc_f_cols)   # encoder df
    stored_args['f_sz'] = len(f_cols)           # featrue size
    stored_args['v2_sz'] = len(value2_cols)     # value 2 size
    stored_args['enc_f2_sz'] = len(enc_f2_cols) # encoder f2 size
    stored_args['f2_sz'] = len(f2_cols)         # feature2 size
    
    
    # calculate [df_list, mask_list]
    # df_list: list of values, preprocessed from DataPreprocessor
    # mask_list: list of mask 
    if args['paddlepaddle_model'] in ['LSTM_enc_f', 'LSTM_enc_f2', 'LSTM_enc2_tp2', 'RNN_multiple']:
        
        if args['mode'] == 'train':             # split data into train and val
            val_size = args['val_days'] * 144
            train_size = total_size - val_size
            train_values = values.iloc[:train_size]
            val_values = values.iloc[train_size-args['input_len']:]
            train_mask = mask.iloc[:train_size]
            val_mask = mask.iloc[train_size-args['input_len']:]
            
            df_list = [train_values, val_values]
            mask_list = [train_mask, val_mask]
            
        elif args['mode'] == 'val': # split data into train, val, and test
            val_days = args['val_days']
            test_days = args['test_days']
            val_size = val_days * 144
            test_size = test_days * 144
            train_size = total_size - val_size - test_size
            # df_list
            train_values = values.iloc[:train_size]
            val_values = values.iloc[train_size-args['input_len']:train_size+val_size]
            test_values = values.iloc[train_size+val_size-args['input_len']:]
            train_mask = mask.iloc[:train_size]
            val_mask = mask.iloc[train_size-args['input_len']:train_size+val_size]
            test_mask = mask.iloc[train_size+val_size-args['input_len']:]
            
            df_list = [train_values, val_values, test_values]
            mask_list = [train_mask, val_mask, test_mask]
        
        elif args['mode'] == 'test': # only test
            test_days = args['test_days']
            test_size = test_days * 144
            test_values = values.iloc[-test_size-args['input_len']:]
            test_mask = mask.iloc[-test_size-args['input_len']:]
            df_list = [test_values]
            mask_list = [test_mask]
            
        elif args['mode'] == 'finetune':  # online finetune
            df_list = [values]
            mask_list = [mask]
            
        elif args['mode'] == 'submit': # only submit 
            if submit_return:
                return (values, mask, value_cols, enc_f_cols, f_cols, value2_cols, enc_f2_cols, f2_cols, extend_features)
            
            x_input = values.iloc[-args['input_len']:]
            x_mask = mask.iloc[-args['input_len']:]   
            df_list = [x_input]
            mask_list = [x_mask]
            
        # calculate scalers
        if args['mode'] in ['train', 'val'] and not args['adaptive_norm']:
            scaler = get_scaler(args, train_values[value_cols].values) # scaler of patv of turbines and turbine groups
            
            scaler2_cols = enc_f_cols +  enc_f2_cols + value2_cols    
            scaler2_cols = [col for col in scaler2_cols if not (col.find('sin')>=0 or col.find('cos')>=0 or col.find('mask_mean')>=0)]
            scaler2 = get_scaler2(args, train_values[scaler2_cols].values)  # caler of other columns
            
            args['stored_args']['stored_scaler'] = scaler
            args['stored_args']['stored_scaler2'] = scaler2
        elif not args['adaptive_norm']:
            scaler = get_scaler(args)
            scaler2 = get_scaler2(args)
        else:
            scaler, scaler2 = None, None
            
        
        # calculate datasets and dataloaders
        datasets = [MyDataset(df_i, mask_i, 
                    value_cols, enc_f_cols, f_cols, value2_cols, enc_f2_cols, f2_cols, args, scaler, 
                    scaler2, extend_features) for df_i, mask_i in zip(df_list, mask_list)]
        
        dataloaders = create_data_loader(datasets, args)
        return dataloaders, datasets


class MyDataset(Dataset):
    """
    Desc: create dataset
    """
    def __init__(self, df, mask, value_cols, enc_f_cols, f_cols, value2_cols, enc_f2_cols,
                 f2_cols, args,  scaler=None, scaler2=None, extend_features=None):
        """
        df: multi-variate preprocessed values
        mask: 
        value_cols, enc_f_cols, f_cols, value2_cols, enc_f2_cols, f2_cols: columns in df
        """
        super().__init__()
        self.args = args
        self.mode = args['mode']
        self.scaler = scaler
        self.scaler2 = scaler2
        self.df = df.copy()
        del df
        self.value_cols = value_cols
        self.enc_f_cols = enc_f_cols
        self.f_cols = f_cols
        self.value2_cols = value2_cols
        self.enc_f2_cols = enc_f2_cols
        self.f2_cols = f2_cols
        
        # standarlization of data
        if self.scaler and not self.args['adaptive_norm']:
            self.df[value_cols] = scaler.transform(self.df[value_cols].values)
        
        if self.scaler2 and not self.args['adaptive_norm']:
            scaler2_cols = enc_f_cols + enc_f2_cols + value2_cols
            scaler2_cols = [col for col in scaler2_cols if not (col.find('sin')>=0 or col.find('cos')>=0 or col.find('mask_mean')>=0)]
            self.df[scaler2_cols] = scaler2.transform(self.df[scaler2_cols].values)
        
        self.mask = mask.values
        self.input_len = args['input_len']
        self.output_len = args['output_len']
        
        self.extend_features = extend_features.values if extend_features is not None else None  # ['sim_tm', 'cos_tm'] as features
        
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.input_len
        df_enc = self.df.iloc[s_begin:s_end].copy()  # encoder part
        
        if self.args['adaptive_norm']:
            ## 
            v = df_enc[self.args['base_nodes']].values.reshape(-1)
            base_mean, base_std = v.mean(), v.std()
            base_mean = np.array([base_mean]*134)
            base_std = np.array([base_std]*134)
            Su = self.args['stored_args']['Su']
            upper_mean = np.dot(Su, base_mean)
            upper_cols = [col for col in self.value_cols if col not in self.args['base_nodes']]
            upper_std = df_enc[upper_cols].values.std(axis=0)
            
            mean = np.hstack([base_mean, upper_mean])
            std = np.hstack([base_std, upper_std])
            scaler = MultiScaler(self.args)
            scaler.mean = mean
            scaler.std = std
            self.args['stored_args']['stored_scaler'] = scaler
            df_enc[self.value_cols] = scaler.transform(df_enc[self.value_cols].values)
            
            # other cols
            enc_o_cols = self.f_cols + self.enc_f_cols + self.f2_cols + self.enc_f2_cols + self.value2_cols
            
            enc_o_cols = [col for col in enc_o_cols if col.find('sin')<0 and col.find('cos')<0 and col.find('mask_mean')<0] #
            o_mean = df_enc[enc_o_cols].values.mean(axis=0)
            o_std = df_enc[enc_o_cols].values.std(axis=0)
            df_enc[enc_o_cols] = (df_enc[enc_o_cols]-o_mean)/o_std   # 
            
        x_enc = df_enc[self.f_cols+self.enc_f_cols + self.value_cols].values
        mask_enc = self.mask[s_begin:s_end]
        x2_enc = df_enc[self.f2_cols + self.enc_f2_cols + self.value2_cols].values
        
        # for submission, only encoder is needed
        if self.mode == 'submit':
            if self.args['adaptive_norm']:
                return x_enc, x2_enc, self.extend_features, mask_enc, mean,std 
            return x_enc, x2_enc, self.extend_features, mask_enc 
        
        # calculate decoder
        r_begin = index + self.input_len
        r_end = r_begin + self.output_len
        df_dec = self.df.iloc[r_begin:r_end].copy()
        
        if self.args['adaptive_norm']:
            df_dec[self.value_cols] = scaler.transform(df_dec[self.value_cols].values)
            df_dec[enc_o_cols] = (df_dec[enc_o_cols]-o_mean)/o_std  
        
        fture_dec = df_dec[self.f_cols].values.astype('float32')
        value_dec = df_dec[self.value_cols].values
        fture2_dec = df_dec[self.f2_cols].values.astype('float32')
        value2_dec = df_dec[self.value2_cols].values
        mask_dec = self.mask[r_begin:r_end]
        
        return x_enc, mask_enc, x2_enc, fture_dec, value_dec, fture2_dec, value2_dec, mask_dec
    
    def __len__(self):
        if self.mode != 'submit':
            return len(self.df) - self.input_len - self.output_len + 1
        return 1


def create_data_loader(datasets, args):
    """
    create dataloaders from list of datasets
    create dataloaders
    datasets: list of dataset
    return: list of dataloaders
    """
    n = len(datasets)
        
    return    [DataLoader(
                        datasets[i],
                        batch_size=args["batch_size"],
                        shuffle=args['shuffle'],
                        num_workers=args["num_workers"],
                        drop_last=args['drop_last'],
                        use_shared_memory=args['use_shared_memory'])
               for i in range(n)
              ]
        
    

def get_scaler(args, train_data=None):
    """
    scaler of turbine and turbine group patv
    """
    if 'stored_scaler' in args['stored_args']:
        return args['stored_args']['stored_scaler']
    
    assert train_data is not None
    if args['scale_method'] == 'multi_scaler':
        scaler = MultiScaler(args)
        scaler.fit(train_data)
        return scaler

def get_scaler2(args, train_data=None):
    """
    scaler of other columns
    """
    
    if 'stored_scaler2' in args['stored_args']:
        return args['stored_args']['stored_scaler2']
    
    assert train_data is not None
    if args['scale_method'] == 'multi_scaler':
        scaler = MultiScaler(args)
        scaler.fit(train_data)
        return scaler

def get_enc_f_scaler(args, train_data=None):
    """
    scaler of enc_f_scaler
    """
    if 'enc_f_scaler' in args['stored_args']:
        return args['stored_args']['enc_f_scaler']
    assert train_data is not None
    if args['scale_method'] == 'multi_scaler':
        scaler = MultiScaler(args)
        scaler.fit(train_data)
        return scaler

def get_enc_f2_scaler(args, train_data=None):
    """
    scaler of enc_f_scaler
    """
    if 'enc_f2_scaler' in args['stored_args']:
        return args['stored_args']['enc_f2_scaler']
    assert train_data is not None
    if args['scale_method'] == 'multi_scaler':
        scaler = MultiScaler(args)
        scaler.fit(train_data)
        return scaler

class MultiScaler(object):
    """
    Desc: Normalization utilities
    """
    def __init__(self, args):
        self.args = args
        self.mean = 0.
        self.std = 1.
        
    
    def fit(self, data, axis=0):
        # type: (paddle.tensor) -> None
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        
        if isinstance(data, paddle.Tensor):
            data = data.detach().numpy()
        self.mean = np.mean(data, axis=axis)
        self.std = np.std(data, axis=axis)
        
    
    def transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        
        mean = paddle.to_tensor(self.mean).type_as(data).to(data.device) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std).type_as(data).to(data.device) if paddle.is_tensor(data) else self.std
        
        return (data - mean) / std
        
    def inverse_transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Restore to the original data
        Args:
            data: the transformed data
        Returns:
            The original data
        """
        
        mean = paddle.to_tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std) if paddle.is_tensor(data) else self.std
        data = (data * std)  + mean
        return data