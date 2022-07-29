import os
import time
import gc
import paddle.nn as nn

from rnn.src.common import *
from rnn.src.paddlepaddle.tools import *
from rnn.src.paddlepaddle.model import *
from rnn.src.paddlepaddle import data_lstm_enc_f
import paddle

def load_trained_model_wrapper(args):
    """
    load trained model_wrapper
    """
    return ModelWrapper(args, load_model(args))

def load_model(args):
    """
    create a model based on args, and if the model is trained, load trained model
    """
    if args['paddlepaddle_model'] == 'LSTM_enc_f':
        model = LSTM_enc_f(args)
        
    elif args['paddlepaddle_model'] == 'LSTM_enc_f2':
        model = LSTM_enc_f2(args)
    
    elif args['paddlepaddle_model'] == 'LSTM_enc2_tp2':
        model = LSTM_enc2_tp2(args, args['stored_args']['A'])#, scaler.std, scaler.mean)
    
    # if model is trained, load the trained model
    model_file = cal_model_file(args)
    
    if os.path.isfile(model_file):
        model_state_dic = paddle.load(model_file)
        model.set_dict(model_state_dic)
    
    return model

def train(args, df=None, model=None):
    """
    train a model
    """
    if args['mode'] not in ['train', 'val', 'finetune']:
        print("mode is not train, val or finetune, exiting....")
        return
    if args['paddlepaddle_model'] in ['LSTM_enc_f', 'LSTM_enc_f2', 'LSTM_enc2_tp2']:
        train_LSTM_enc_f(args, df, model)
    

def train_LSTM_enc_f(args, df=None, model=None):
    """
    train a model, the model should be from ['LSTM_enc_f', 'LSTM_enc_f2', 'LSTM_enc2_tp2']
    """
    tic = time.time()
    if args['mode'] not in ['train', 'val', 'finetune']:
        print("mode is not train, val or 'finetune', exiting....")
        return
    
    dataloaders, datasets = data_lstm_enc_f.load_data(args, df) 
    
    train_loader = dataloaders[0]
    if args['mode'] not in ['finetune', 'submit']:
        val_loader = dataloaders[1]
        test_loader = dataloaders[-1]
    if model is None:
        model = load_model(args)
    
    model_file = cal_model_file(args)
    criterion = nn.MSELoss(reduction='mean')
    early_stopping = EarlyStopping(args, delta=0)
    #model.eval()
    clip = paddle.nn.ClipGradByNorm(clip_norm=50.0)
    model_optim = paddle.optimizer.Adam(parameters=model.parameters(),
                                    learning_rate=args['lr'],
                                    grad_clip=clip)
    iter_count = 0
    train_loss = []
    start_time = time.time()
    epochs = args['train_epochs'] if args['mode'] != 'finetune' else args.get('finetune_epochs', 2)
    
    for epoch in range(epochs):
        print('epoch:', epoch)
        if args['mode']=='finetune' and time.time() - tic > 150:
            print("enough time for finetuning this instance, iter_count=", iter_count)
            del dataloaders
            del datasets
            gc.collect()
            break
        adjust_learning_rate(model_optim, epoch, args)  # decreasing learning rate after each eopoch
        model_optim = paddle.optimizer.Adam(parameters=model.parameters(),
                                    learning_rate=args['lr'],
                                    grad_clip=clip)
        
        for x_enc, mask_enc, x2_enc, fture_dec, value_dec, fture2_dec, value2_dec, mask_dec in train_loader:
            x_enc = x_enc.astype('float32')
            x2_enc = x2_enc.astype('float32')
            if fture_dec.shape[-1] > 0:
                fture_dec = fture_dec.astype('float32')
            value_dec = value_dec.astype('float32')
            if fture2_dec.shape[-1] > 0:
                fture2_dec = fture2_dec.astype('float32')
            value2_dec = value2_dec.astype('float32')
            mask_dec = mask_dec.astype('float32')
            
            iter_count += 1
            
            model.train()
            
            if args['paddlepaddle_model'] == 'LSTM_enc2_tp2':  # coherent forecast, need to know to normalization parameters
                scaler = args['stored_args']['stored_scaler']
                u, std = scaler.mean, scaler.std
                value_pred, value2_pred = model(u, std, x_enc, x2_enc, fture_dec, fture2_dec)
            else:
                value_pred, value2_pred = model(x_enc, x2_enc, fture_dec, fture2_dec)
            
            if args['masked_loss']: # whether to use masked loss: if true, only loss of nomal values will be counted
                value_dec *= mask_dec
                value_pred *= mask_dec
            loss = criterion(value_pred, value_dec)      
            loss2 = criterion(value2_pred, value2_dec)
            loss2_w = args['loss2_w']*0.9**epoch
            loss_w = 1 - loss2_w   
            #total_loss = args['loss_w']*loss + args['loss2_w']*loss2
            total_loss = loss_w*loss + loss2_w*loss2  # combined loss of two LSTM layers, weight of second layer will decrease
            train_loss.append(loss.item())
            model_optim.clear_grad()
            
            total_loss.backward()
            model_optim.minimize(loss)
            model_optim.step()
            
            gc.collect()
            if args['mode']=='finetune' and time.time() - tic > 150:
                print("enough time for finetuning this instance, iter_count=", iter_count)
                break
            
            if iter_count % 128 == 127 and args['mode']!='finetune':
                train_loss = np.mean(train_loss)
                model.eval()
                val_loss, val_score = val_lstm_enc_f(model, val_loader, args, kdd_score=True)  
                if args['valid_type'] == 1:
                    early_stopping(val_loss, model, model_file)
                else:
                    early_stopping(val_score, model, model_file)
                
                test_loss, test_score = val_lstm_enc_f(model, test_loader, args, kdd_score=True) 
                
                
                end_time = time.time()
                print('train_loss=', train_loss, 'val_loss=', val_loss, 'val_score', val_score, 'test_score',  test_score, 'time:', end_time-start_time)
                
                if early_stopping.early_stop:
                    print("early stopping criteria encoutered, stop")
                    
                    return
                start_time = time.time()
                train_loss = []
                

class ModelWrapper():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.model.eval()
        self.args['finetune'] = False   # online memeory error, disable this 
        
    def fine_tune(self, df, model):
        mode = self.args['mode']
        self.args['mode'] = 'finetune'
        
        train(self.args, df, model)
    
        self.args['mode'] = mode
    
    def predict(self, df_encoder):
        """
        df_encoder: raw df, similar to what returns from pd.read_csv(args['data_file'])
        """
        if self.args.get('finetune', False):
            self.model = load_model(self.args)
            self.fine_tune(df_encoder, self.model)
            print("done with fintue")
            #gc.collect()
        
        print("forecasting")
        self.model.eval()
        if self.args['paddlepaddle_model'] in ['LSTM_hr_coherent_tp2', 'LSTM_hr']:  # these models are no longer used 
            """
            
            """
            dataloader, dataset = wind_turbine_data.load_submit_data_from_raw(self.args, df_encoder)
            x_enc, fture_y,  mask_x  = list(dataloader[0])[0]
            x_enc = x_enc.astype('float32')
            fture_y = fture_y.astype('float32')
            
            y = self.model(x_enc, feature_dec=fture_y, train=False)
            y = self.args['stored_args']['stored_scaler'].inverse_transform(y).detach().numpy()[:,:,:134].clip(0, self.args['max_turbine_patv'])
            
            return np.einsum('ijk->kji', y).clip(0, self.args['max_patv'])
        
        elif self.args['paddlepaddle_model'] in ['LSTM_enc_f', 'LSTM_enc_f2', 'LSTM_enc2_tp2']:
            mode = self.args['mode']
            self.args['mode'] = 'submit'
            dataloaders, datasets = data_lstm_enc_f.load_data(self.args, df_encoder)
            if self.args['adaptive_norm']:
                x_enc, x2_enc, extend_features, mask_enc, mean, std = list(dataloaders[0])[0]
                mean = mean.detach().numpy().reshape(-1)[0]
                std = std.detach().numpy().reshape(-1)[0]
            else:
                x_enc, x2_enc, extend_features, mask_enc = list(dataloaders[0])[0]
            x_enc = x_enc.astype('float32')
            x2_enc = x2_enc.astype('float32')
            extend_features = extend_features.astype('float32')
            if len(self.args['f_cols'])>0:
                f_dec = extend_features
            else:
                f_dec = None
                
            if len(self.args['f2_cols']) > 0:
                f2_dec = extend_features  # ['sin_tm', 'cos_tm']
            else:
                f2_dec = None
            
            if self.args['paddlepaddle_model'] == 'LSTM_enc2_tp2':  # with hierarchical coherecy constraints
                scaler = self.args['stored_args']['stored_scaler']
                u, std = scaler.mean, scaler.std
                y, _ = model(u, std, x_enc, x2_enc, f_dec=f_dec, f2_dec=f2_dec)
            else:
                y, _ = self.model(x_enc, x2_enc, f_dec=f_dec, f2_dec=f2_dec)
            
            
            if self.args['adaptive_norm']:
                y = y.detach().numpy()[:,:,:134]*std+mean
            else:
                y = self.args['stored_args']['stored_scaler'].inverse_transform(y).detach().numpy()[:,:,:134]
            
            
            y *= self.args['max_patv']
            y = y.clip(0, self.args['max_patv'])
            self.args['mode'] = mode
            y = np.einsum('ijk->kji', y)
            del dataloaders, datasets, x_enc, x2_enc, extend_features, mask_enc
            if self.args.get('finetune', False):
                del self.model
            gc.collect()
            paddle.device.cuda.empty_cache()
            return y
            
    
