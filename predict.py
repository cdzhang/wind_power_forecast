import numpy as np
from prepare import prep_env as global_prep_env
import sys, os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print(path, 'added')

def remove_path(path):
    if path in sys.path:
        sys.path.remove(path)

global_args = global_prep_env()
model_dirs = global_args['model_dirs']   # 

model_args = []
model_forecasts = []

path = os.path.dirname(os.path.abspath(__file__))

for model_dir in model_dirs:
    
    model_path = os.path.join(path, model_dir)
    add_path(model_path)
    model_path_src = os.path.join(model_path, 'src')
    if model_dir == 'rnn':
        add_path(model_path_src)
    
    arg_model = '{}.src.prepare'.format(model_dir)
    model_arg = __import__(arg_model).src.prepare.prep_env()
    model_args.append(model_arg)
    
    model_predict = '{}.src.predict'.format(model_dir)
    model_predict = __import__(model_predict).src.predict.forecast
    model_forecasts.append(model_predict)
    
    remove_path(model_path)
    remove_path(model_path_src)

def forecast(settings):
    
    yhats = []
    for model_arg, model_forecast in zip(model_args, model_forecasts):
        model_arg['path_to_test_x'] =  settings['path_to_test_x']
        yhat = model_forecast(model_arg)
        yhats.append(yhat)
    yhat = yhats[0].copy()
    #yhat[:,144:,:] = yhats[0][:,144:,:]*0.5+0.5*yhats[1][:,144:,:]
    yhat[:,200:,:] = yhats[0][:,200:,:]*0.5+0.5*yhats[1][:,200:,:]
    return yhat
    #return yhats
    #return np.concatenate(yhats, axis=2).mean(axis=-1, keepdims=True)
