import os
import sys
import traceback
import sys
import re
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print(path, 'added')

path = os.path.dirname(os.path.abspath(__file__))
p1_path = os.path.join(path, '../..')
add_path(p1_path)
add_path(path)
add_path(path+'/prepares')

## local pacakges, ignore this
add_path('/home/notebook/code/group/intention_rec/TimeSeries/kdd2022/paddle_lc_libs')

#from prepare import *
if len(sys.argv) < 2:
    module_file = 'prepare.py'
else:
    module_file = sys.argv[1]

module_file = re.sub('.py$', '', module_file)
print(module_file)
module = __import__(module_file)

from common import save_args

args = module.prep_env()
args['mode'] = 'val'


path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if path not in sys.path:
    sys.path.insert(0, path)
module = __import__(args['model_dir'])

module.train(args)

save_args(args)

#print("model trained, please set 'mode' in prep_env to 'submit'")
#os.system("python3 pack.py {}".format(module_file))
