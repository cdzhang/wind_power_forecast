import sys
import os
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print(path, 'added')

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.abspath(os.path.join(path, '../..'))
add_path(path)

import paddle
from rnn.src.prepare import *
args = prep_env()

if paddle.device.is_compiled_with_cuda() and args['use_gpu']:
    try:
        paddle.device.set_device('gpu:0')
    except:
        paddle.device.set_device('cpu')
else:
    paddle.device.set_device('cpu')
#print("device is", paddle.get_device())


from rnn.src.paddlepaddle.model_wrapper import train, load_trained_model_wrapper, ModelWrapper
