import pandas as pd
import numpy as np
from rnn.src.hierarchy.hierarchical_dicts import *
import lightgbm as lgb
import pickle, os
#from fbprophet import Prophet

class DataPreprocessor():
    """
    common data preprocessing methods
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.raw_data = None
        self.loc = None
        self.hierarchy_dic = {}
        
    def __call__(self, df=None):
        """
        return the preprocessed data
        """
        if self.args['preprocess_method'] == 'multi_var':
            return self.process_data_multi_variate(df)
    
    def process_data_multi_variate(self, df=None, return_df=False):
        """
        multi-variate time series preprocessing
        """
        if df is None:
            df = self.read_hist_data() 
        df = self.preprocessing(df)
        
        if self.args['generate_noisy_sample']:  # generate sample according to wind speed
            df = self.generate_noisy_sample(df, self.args['normal_noise'])
        
        wt = self.process_field_tmp(df)
        wd = self.process_field_wspd(df)
        df = df.merge(wt, how='left').merge(wd, how='left')
        
        df['tm'] = df['Day'].astype(str) + ',' + df['Tmstamp']
        
        values = pd.pivot_table(index='tm', columns=['TurbID'], values='Patv',
                 data=df, dropna=False).reset_index()
        mask = pd.pivot_table(index='tm', columns=['TurbID'], values='mask', data=df).reset_index()
        tm = values['tm'].str.split(',').str
        values['Tmstamp'] = tm[1]
        values['Day'] = tm[0].astype('float32').astype(int)
        
        mask['Tmstamp'] = tm[1]
        mask['Day'] = tm[0].astype('float32').astype(int)
        values = values.sort_values(['Day', 'Tmstamp']) # 重要
        mask = mask.sort_values(['Day', 'Tmstamp'])
        values['mask_mean'] = mask.mean(axis=1)
        values = values.fillna(method='bfill').fillna(method='ffill')
        
        
        if 'hierarchical' in self.args and self.args['hierarchical']:
            # calculate upper nodes
            base_nodes, hie_dic = self.args['base_nodes'], self.args['hie_dic']
            Su, A, upper_nodes, all_nodes = hierarchy_dict_to_matrix(base_nodes, hie_dic)
            self.hierarchy_dic['A'] = A
            self.hierarchy_dic['all_nodes'] = all_nodes
            for upper_node in upper_nodes:
                values[upper_node] = values[hie_dic[upper_node]].sum(axis=1)
                if self.args.get('hierarchical_mask', False):
                    mask[upper_node] = mask[hie_dic[upper_node]].min(axis=1)
            value_cols = all_nodes
            self.args['stored_args']['A'] = A
            self.args['stored_args']['Su'] = Su
        else:
            value_cols = self.args['base_nodes']
        if self.args.get('hierarchical_mask', False):
            mask = mask[value_cols]
        else:
            mask = mask[self.args['base_nodes']]
        mask = self.set_to_float32(mask)
        values = values.merge(wt, how='left').merge(wd, how='left').fillna(method='bfill').fillna(method='ffill')
        
        
        # calculate feature cols
        values['min_of_day'] = pd.to_datetime(values['Tmstamp']).dt.hour*60 + pd.to_datetime(values['Tmstamp']).dt.minute
        values['minutes'] = values['min_of_day'] + (values['Day']-1)*1440
        
        values['sin_tm'] = np.sin(2*np.pi*values['min_of_day'] / 1440)
        values['cos_tm'] = np.cos(2*np.pi*values['min_of_day'] / 1440)
        values['day_norm'] = values['Day'] / 200
        
        # median columns
        median_cols = self.args.get('more_median_cols', [])
        for median_col in median_cols:
            median_cols = self.args['more_median_cols']
            medians = self.cal_median(df, median_cols)
            values = values.merge(medians, how='left')
        
        values = self.set_to_float32(values)
        extend_features = None
        if self.args['mode'] == 'submit':  # feature_cols 
            extend_features = values[['sin_tm', 'cos_tm']].iloc[-288:].copy()
            extend_features.reset_index().drop('index', axis=1).iloc[:self.args['output_len']]
        feature_cols = ['sin_tm', 'cos_tm']
        if return_df:
            return values, mask, value_cols, extend_features, df
        return values, mask, value_cols, extend_features
        
    def preprocessing(self, df):
        """
        data preprocessing
        """
        if self.args['envelope_abnormal']: # whether to use max patv - wspd curve to detect abnormal values
            df = self.cal_envelope(df)
        df = self.fill_missing_tms(df) # fill in missing ['Day', 'Tmstamp']
        df = df.sort_values(['Day', 'Tmstamp', 'TurbID'])
        
        df = self.adjust_wind_angle(df) # Adjust wind direction
        df = self.cal_mask(df)
        df = self.adjust_Etmp_Itmp(df)  # Adjust Etmp and Itmp
        df['Wspd'] = df['Wspd'].fillna(method='bfill').fillna(method='ffill')
        if not self.args['debug']: # delete unused columns to reduce memory
            cols = ['TurbID', 'Day', 'Tmstamp', 'Wspd', 'Etmp', 'Itmp', 'Patv', 
            'w_angle_sin', 'w_angle_cos', 'mask', 'pab_mask',  'Etmp_median',  'Itmp_median']
            for col in df.columns:
                if col not in cols:
                    del df[col]

        return self.set_to_float32(df)
    
    
    def generate_noisy_sample(self, df, normal_noise=False):
        """
        df: should contain mask column
        normal_noise: should add noise to normal samples
        """
        if 'mask' not in df.columns:
            df = self.cal_mask(df)
        
        df['ws'] = df['Wspd']//0.5 * 0.5
        i1 = df['Wspd'] <= 2.5
        i2 = df['Wspd'] >= 10

        df.loc[i1, 'ws'] = 0
        df.loc[i2, 'ws'] = 10
        path = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(path)!='src':
            path = os.path.abspath(os.path.join(path, '..'))
        noisy_file = os.path.join(path, "stored_process/noisy_sample.pkl") # previously stored normal samples, 1000 sample for every wspd range
        with open(noisy_file, 'rb') as f:
            dic_sample = pickle.load(f)
        data = []
        df1 = df.query("mask==1")  # normal samples
        for ws, dfw in df.query("mask==0").groupby('ws'):
            df0 = dfw.copy()
            v1 = dic_sample[ws].astype('float32')
            df0['Patv'] = np.random.choice(v1, len(df0))
            data.append(df0)
        data.append(df1)
        data = pd.concat(data).sort_values(['TurbID', 'Day', 'Tmstamp'])
        return data

    def process_field_tmp(self, df_raw):
        """
        calculate wind farm tmmperature statistics
        """
        tmp = df_raw.groupby(['Day', 'Tmstamp'])['Etmp'].agg([np.mean, 
               np.median, np.std]).reset_index().fillna(method='bfill')
        tmp.columns = ['Day', 'Tmstamp', 'etmp_mean', 'etmp_median', 'etmp_std']
        return tmp
    
    def process_field_wspd(self, df_raw):
        """
        calculate wind farm wind speed statistics
        """
        ws = df_raw.groupby(['Day', 'Tmstamp'])['Wspd'].agg([np.mean, 
               np.median, np.std]).reset_index().fillna(method='bfill')
        ws.columns = ['Day', 'Tmstamp', 'ws_mean', 'ws_median', 'ws_std']
        return ws
        
    def adjust_wind_angle(self, df):
        """
        adjust wind directions
        """
        dfa = pd.read_csv(self.args['angle_diff_file'])
        df = df.merge(dfa, how='left')
        df['w_angle'] = df['Ndir'] + df['Wdir']
        df['w_angle_adj'] = df['w_angle'] + df['diff_angle']
        df['w_angle_sin'] = np.sin(df['w_angle_adj'] * np.pi/180).fillna(method='bfill').fillna(method='ffill').astype('float32')
        df['w_angle_cos'] = np.cos(df['w_angle_adj'] * np.pi/180).fillna(method='bfill').fillna(method='ffill').astype('float32')
        return df

    
    def fill_missing_tms(self, df_raw):
        """
        fill in missing Day & Tmstamp
        """
        tms = df_raw[['Day', 'Tmstamp']].drop_duplicates().sort_values(['Day', 'Tmstamp'])
        day_start, tm_start = tms['Day'].iloc[0], tms['Tmstamp'].iloc[0]
        day_end, tm_end = tms['Day'].iloc[-1], tms['Tmstamp'].iloc[-1]
        start_date = pd.to_datetime('2000-01-01 ' + tm_start) 
        end_date = pd.to_datetime('2000-01-01 ' + tm_end) + pd.Timedelta(days=day_end-day_start)
        dfd = pd.DataFrame(pd.date_range(start_date, end_date, freq='10min'), columns=['tm'])
        dfd['Day'] = (pd.to_datetime(dfd['tm'].dt.date) - pd.to_datetime('2000-01-01')).dt.days + day_start
        dfd['Tmstamp'] = dfd['tm'].dt.strftime('%H:%M')

        turbs = df_raw[['TurbID']].drop_duplicates()
        dfd['xxxyyyy'] = 1
        turbs['xxxyyyy'] = 1
        dfd = turbs.merge(dfd).drop(['tm', 'xxxyyyy'], axis=1)
        return dfd.merge(df_raw, how='left')
    
    def cal_mask(self, raw_df):
        """
        cal mask
        """
        df = raw_df.copy()
        
        s = (df['Patv'] <= 0) | (df['Patv'] <= 0) & (df['Wspd'] > 2.5) | \
        (df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89) | \
        (df['Wdir'] < -180) | (df['Wdir'] > 180) | (df['Ndir'] < -720) | (df['Ndir'] > 720) | \
        (pd.isna(df['Patv']))                          # official unknown
        df['mask'] = 1-s
        df.loc[df['mask']==0,'Patv'] = np.nan
        df['pab_mask'] = 1 - ((df['Pab1'] > 89) | (df['Pab2'] > 89) | (df['Pab3'] > 89))  # abnormal pab values
        
        return df
    
    def cal_envelope(self, raw_df):
        """
        use max patv - wspd curve to adjust data
        f(Wspd) = a*Wspd**beta + b*Wspd if f(Wspd) < mx, else wx
        """
        ev_paras = pd.read_csv(self.args['envelope_file'])
        df = raw_df.merge(ev_paras)
        df['f1'] = df['a']*df['Wspd']**df['beta'] + df['b']*df['Wspd']
        df['f'] = np.minimum(df['mx'], df['f1'])
        df['mask2'] = df['f']<df['Patv']
        if self.args['envelope_abnormal']:  
            df['origin_Patv'] = df['Patv']
            df.loc[df['mask2'], 'Patv'] = df.loc[df['mask2'], 'f']
        return df
    
    def adjust_Etmp_Itmp(self, raw_df, ub=85, lb=15):
        """
        Some turbines might have broken thermometor, adjust etmp and itmp
        """
        f1 = lambda x:np.nanpercentile(x, ub)
        f2 = lambda x:np.nanpercentile(x, lb)
        raw_df = raw_df.copy()
        raw_df['origin_Etmp'] = raw_df['Etmp']
        raw_df['origin_Itmp'] = raw_df['Itmp']
        
        x = raw_df.query("-20<=Etmp<=50").groupby(['Day', 'Tmstamp'])[['Etmp', 'Itmp']].agg([np.median, f1, f2]).reset_index()
        x.columns = ['Day', 'Tmstamp', 'Etmp_median', 'Etmp_ub', 'Etmp_lb', 'Itmp_median', 'Itmp_ub', 'Itmp_lb']
        
        y = raw_df.merge(x, how='left')
        index1 = (y['Etmp'] > y['Etmp_ub']) | (y['Etmp'] < y['Etmp_lb']).values
        y.loc[index1, 'Etmp'] = y.loc[index1, 'Etmp_median']
        
        index2 = (y['Itmp'] > y['Itmp_ub']) | (y['Itmp'] < y['Itmp_lb']).values
        y.loc[index2, 'Itmp'] = y.loc[index2, 'Itmp_median']
        
        return y
    
    def lgb_abnormal_adjust(self, raw_df):
        """
        use lgb to fill in unknown patv
        """
        df = raw_df.copy()
        if 'lgb_model' not in self.args:
            self.args['lgb_model'] = lgb.Booster(model_file=self.args['abnormal_lgb_file'])
        dt = pd.to_datetime(df['Tmstamp'])
        df['min_of_day'] = dt.dt.hour*60 + dt.dt.minute

        df['sin_tm'] = np.sin(2*np.pi*df['min_of_day'] / 1440)
        df['cos_tm'] = np.cos(2*np.pi*df['min_of_day'] / 1440)
        itmp_median = df.groupby(['Day', 'Tmstamp'])['Itmp'].agg(np.median).reset_index()
        itmp_median.columns = ['Day', 'Tmstamp', 'Itmp_median']
        df = df.merge(itmp_median, how='left')
        mean_df = df.query("mask==1 and mask2==False").groupby(['Day', 'Tmstamp'])[['Patv', 'etmp_median', 'Itmp_median', 'Wspd']].agg(np.mean).reset_index()
        
        
        mean_df.columns = ['Day', 'Tmstamp', "Patv_mean", "Etmp_median", "Itmp_median", 'Wspd_mean']
        
        mask_mean = df.groupby(['Day', 'Tmstamp'])['mask'].mean().reset_index()
        mask_mean.columns = ['Day', 'Tmstamp', 'mask_mean']
        data = df[['Day', 'Tmstamp', 'mask', 'mask2', 'Wspd', 'TurbID', 'sin_tm', 'cos_tm', 'Etmp', 'Itmp', 'Wdir', 'Patv', 'mx', 'cp', 'f']].merge(mean_df, how='left').merge(mask_mean, how='left')
        
        feature_cols = ['TurbID', 'sin_tm', 'cos_tm', 'Etmp', 'Itmp', 'Wspd',
                'Wdir', 'Patv_mean', 'Etmp_median', 'Itmp_median', 'Wspd_mean', 'mask_mean']
        
        index = data['mask'] == 0
        X = data.loc[index][feature_cols].copy()
        yhat = self.args['lgb_model'].predict(X)
        df.loc[index, 'Patv'] = yhat
        
        df.loc[df['Patv']<0, 'Patv'] = 0
        df.loc[df['Patv']>df['f'], 'Patv'] = df.loc[df['Patv']>df['f'], 'f']
        
        return df
        
    def cal_median(self, raw_df, cols):
        """
        calculate the median values(median of turbines)
        """
        if type(cols) == str:
            cols = [cols]
        x = None
        for col in cols:
            ci = raw_df.groupby(['Day', 'Tmstamp'])[col].agg(np.nanmedian).reset_index().fillna(method='bfill')
            ci.columns = ['Day', 'Tmstamp', col + '_median']
            if x is None:
                x = ci
            else:
                x = x.merge(ci)
        return x
    
    
    def read_hist_data(self):
        if self.raw_data is None:
            self.raw_data = pd.read_csv(self.args['data_file'])
            if 'partial_data' in self.args and self.args['mode'] in ['train', 'val'] and self.args['partial_data'] is not None:
                partial_days = self.args['partial_data']
                self.raw_data = self.raw_data.query("Day<=@partial_days").copy()
            self.raw_data = self.set_to_float32(self.raw_data)
            self.raw_data['origin_Patv'] = self.raw_data['Patv']
        return self.raw_data
    
    def set_to_float32(self, df, float_cols=None):
        """
        set numerical columns to float32
        """
        if float_cols is None:
            float_cols = df.select_dtypes(include=np.number).columns.to_list()
        df[float_cols] = df[float_cols].astype('float32')
        return df 

    def read_location_data(self):
        if self.loc is None:
            self.loc = pd.read_csv(self.args['location_file'])
        return self.loc


