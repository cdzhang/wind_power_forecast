import pickle
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.abspath(os.path.join(path,'..'))
sys.path.append(path)

def cal_model_file(args):
    """
    calculate the location of stored model file
    """
    path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(path, args['checkpoints'])
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    model_file = os.path.join(checkpoint_path, args['model_file_name'])
    return model_file

def cal_arg_file(args):
    """
    calculated the location of stored args
    """
    path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(path, args['checkpoints'])
    checkpoint_path = os.path.abspath(checkpoint_path)
    arg_file = os.path.join(checkpoint_path, args['model_file_name']+'updated_args')
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    return arg_file

def save_args(args):
    """
    save updated args
    """
    stored_args = args['stored_args']
    file = cal_arg_file(args)
    with open(file, 'wb') as f:
        pickle.dump(stored_args, f)

def load_args(args):
    """
    load saved args
    """
    file = cal_arg_file(args)
    if os.path.isfile(file):
        with open(file, 'rb') as f:
            args['stored_args'] = pickle.load(f)
    return args


