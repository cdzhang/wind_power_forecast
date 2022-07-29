# -*-Encoding: utf-8 -*-

import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import numpy as np


class LSTM_enc_f(nn.Layer):
    """
    two layers of LSTM
    global wind speed is predicted by one layer of lstm.
    global wind speed is the median speed of all turbines
    """
    def __init__(self, args, final_relu=False):
        # type: (dict) -> None
        """
        Desc:
            __init__
        Args:
            args: a dict of parameters
        """
        super(LSTM_enc_f, self).__init__()
        for key in ['enc_f_sz', 'f_sz', 'enc_f2_sz', 'f2_sz']:
            if key not in args:
                args[key] = 0
        
        patv_enc_in_sz = args['v_sz'] + args['enc_f_sz'] + args['f_sz']  # layer1 encoder input size
        patv_dec_in_sz = args['v_sz'] + args['v2_sz'] + args['f_sz']     # layer1 decoder input size
        wt_in_sz = args['v2_sz'] + args['f2_sz'] + args['enc_f2_sz']     # layer2 size
        
        self.lstm_patv_enc = nn.LSTM(input_size=patv_enc_in_sz, hidden_size=args['hid_sz'])
        
        self.lstm_patv_dec = nn.LSTM(input_size=patv_dec_in_sz, hidden_size=args['hid_sz'])
        
        self.lstm_wt = nn.LSTM(input_size=wt_in_sz, hidden_size=args['hid2_sz'])     # wimd and tmperature
        
        
        
        self.projection = nn.Linear(args['hid_sz'], args['hid_sz'])
        self.linear = nn.Linear(args['hid_sz'], args['v_sz'])
        
        self.projection2 = nn.Linear(args['hid2_sz'], args['hid2_sz'])
        self.linear2 = nn.Linear(args['hid2_sz'], args['v2_sz'])
        
        self.dropout = nn.Dropout(args["dropout"])
        
        self.args = args
        self.pred_len = args['output_len']
        
        self.final_relu = final_relu
    
    def forward(self, x_enc, x2_enc, f_dec=None, f2_dec=None):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc: the input for enc, first RNN
            x2_dec: the input for enc, second RNN
        
            f_dec: the decoder features for first RNN
            f2_dec: the decoder features for second RNN
        Returns:
            two tensors
        """
        bs = x_enc.shape[0]
        ss = x_enc.shape[1]
        
        _, (h,c) = self.lstm_patv_enc(x_enc)  # x(t+1)
        _, (h2, c2) = self.lstm_wt(x2_enc)
        
        y_last = x_enc[:,-1:,-self.args['v_sz']:] 
        y_last2 = x2_enc[:,-1:,-self.args['v2_sz']:] 
        
        ys = []
        ys2 = []
        
        for i in range(self.pred_len):
            feature2 = f2_dec[:,i:i+1,:] if f2_dec is not None else paddle.empty([bs, ss, 0])
            x2i = paddle.concat([feature2, y_last2], axis=2)  # concat of feature2 and value2
            y2i, (h2, c2) = self.lstm_wt(x2i, (h2, c2))       
            y_last2 = self.linear2(fluid.layers.relu(self.projection2(y2i))) 
            ys2.append(y_last2)

            feature = f_dec[:,i:i+1,:] if f_dec is not None else paddle.empty([bs, ss, 0])
            xi = paddle.concat([y_last2, feature, y_last], axis=2)  #  concat
            yi, (h, c) = self.lstm_patv_dec(xi, (h,c))
            y_last = self.linear(fluid.layers.relu(self.projection(yi)))  
            ys.append(y_last)   
        
        ys = paddle.concat(ys, axis=1)
        ys2 = paddle.concat(ys2, axis=1)
        return ys, ys2

class Attention(paddle.nn.Layer):
    def __init__(self, scale):
        super(Attention, self).__init__()
        self.scale = scale 
    def forward(self, Q, K, V):
        attn = paddle.einsum('ijk,ijl->ikl', Q, K)  / self.scale
        attn = paddle.nn.Softmax(axis=-1)(attn)
        attn = paddle.einsum('ikl,ijl->ijk', attn, V)
        return attn
    
class LSTM_enc_f2(nn.Layer):
    """
    add encoder features for second lstm layer
    """
    def __init__(self, args, final_relu=False):
        """
        Desc:
            __init__
        Args:
            settings: a dict of parameters
        """
        super(LSTM_enc_f2, self).__init__()

        self.args = args
        self.stored_args = args['stored_args']
        for key in ['enc_f_sz', 'f_sz', 'enc_f2_sz', 'f2_sz']:
            if key not in self.stored_args:
                self.stored_args[key] = 0
        
        
        patv_enc_in_sz = self.stored_args['v_sz'] + self.stored_args['enc_f_sz'] + self.stored_args['f_sz']  # layer1 encoder input size
        patv_dec_in_sz = self.stored_args['v_sz'] + self.stored_args['v2_sz'] + self.stored_args['f_sz']     # layer1 decoder input size
        
        wt_enc_in_sz = self.stored_args['v2_sz'] + self.stored_args['enc_f2_sz'] + self.stored_args['f2_sz']  # layer2 encoder input size
        wt_dec_in_sz = self.stored_args['v2_sz'] + self.stored_args['f2_sz']         #layer2 decoder input size
        
        
        self.lstm_patv_enc = nn.LSTM(input_size=patv_enc_in_sz, hidden_size=args['hid_sz'])  # layer1 encoder
        self.lstm_patv_dec = nn.LSTM(input_size=patv_dec_in_sz, hidden_size=args['hid_sz'])  # layer1 decoder
        
        self.lstm_wt_enc = nn.LSTM(input_size=wt_enc_in_sz, hidden_size=args['hid2_sz'])     # layer2 encoder
        self.lstm_wt_dec = nn.LSTM(input_size=wt_dec_in_sz, hidden_size=args['hid2_sz'])     # layer2 decoder
        
        
        self.projection = nn.Linear(args['hid_sz'], args['hid_sz'])
        self.linear = nn.Linear(args['hid_sz'], self.stored_args['v_sz'])
        
        self.projection2 = nn.Linear(args['hid2_sz'], args['hid2_sz'])
        self.linear2 = nn.Linear(args['hid2_sz'], self.stored_args['v2_sz'])
        self.dropout = nn.Dropout(args["dropout"])
        
        self.pred_len = args['output_len']
        
        self.final_relu = final_relu
        self.lb = self.args['lstm_dec_lookbk']   # lookback window, calculate the decoder input value
        if args.get('output_attention', False):  # whether add an extra attention layer
            self.attention = Attention(np.sqrt(args['output_len']))
            
    def forward(self, x_enc, x2_enc, f_dec=None, f2_dec=None):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            The specific implementation for interface forward
        Args:
            x_enc: the input for enc, first RNN
            x2_dec: the input for enc, second RNN
        
            f_dec: the decoder features for first RNN
            f2_dec: the decoder features for second RNN
        Returns:
            two tensors
        """
        bs = x_enc.shape[0]
        ss = x_enc.shape[1]
        
        _, (h,c) = self.lstm_patv_enc(x_enc)  # x(t+1)
        _, (h2, c2) = self.lstm_wt_enc(x2_enc)
        
        y_last = x_enc[:,-self.lb:,-self.stored_args['v_sz']:].median(axis=1,keepdim=True) # median of history true value as input for decoder
        y_last2 = x2_enc[:,-self.lb:,-self.stored_args['v2_sz']:].median(axis=1, keepdim=True) 
        
        ys = []
        ys2 = []
        
        for i in range(self.pred_len):  # recurrent prediction
            feature2 = f2_dec[:,i:i+1,:] if f2_dec is not None else paddle.empty([bs, ss, 0])
            x2i = paddle.concat([feature2, y_last2], axis=2)  # concatenation of feature2 and value2
            y2i, (h2, c2) = self.lstm_wt_dec(x2i, (h2, c2))    
            y_last2 = self.linear2(fluid.layers.relu(self.projection2(y2i))) 
            ys2.append(y_last2)

            feature = f_dec[:,i:i+1,:] if f_dec is not None else paddle.empty([bs, ss, 0])

            xi = paddle.concat([y_last2, feature, y_last], axis=2)  #  concat
            yi, (h, c) = self.lstm_patv_dec(xi, (h,c))
            y_last = self.linear(fluid.layers.relu(self.projection(yi)))  
            ys.append(y_last)   # x(t+1)
        
        ys = paddle.concat(ys, axis=1)
        ys2 = paddle.concat(ys2, axis=1)

        if self.args.get('output_attention', False):
            ys += self.attention(ys, ys, ys)
        return ys, ys2
    
    
class LSTM_enc2_tp2(nn.Layer):
    def __init__(self, settings, A):
        """
        Add hierarchical conherence layer
        y is the forecast of all turbines and turbine groups
        y' = coherent(y)
        y' satisfies the coherency constraints: Ay'=0
        y and y' are normalized 
        """
        super(LSTM_enc2_tp2, self).__init__()
        self.lstm_enc = LSTM_enc_f2(settings)
        
        B = np.linalg.inv(np.dot(A,A.T))
        C = np.dot(A.T, B)
        B = np.dot(C, A)
        self.M = np.eye(settings['v_sz']) - B
        self.M = paddle.to_tensor(self.M, dtype=paddle.float32)
        
    def forward(self, u, std, *args, **kwargs):
        std = paddle.to_tensor(std, dtype=paddle.float32)
        u = paddle.to_tensor(u, dtype=paddle.float32)
        y, ys2 = self.lstm_enc(*args, **kwargs)
        y = paddle.einsum('ij,klj->kli', self.M, std*y+u) - u
        y = y / self.std
        
        return y, ys2
    