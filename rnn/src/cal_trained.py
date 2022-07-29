import glob
import os
import datetime
import sys

# change the following to the right path
os.chdir('/home/notebook/code/personal/kdd2022/src_ensemble/rnn/src')

ls = glob.glob('checkpoints/RNN*_hh*updated_args')
#ls = glob.glob('checkpoints/RNN_hh*updated_args')

ready = []
for f in ls:
    x = f.replace('updated_args', '').split('_')
    if f.find('TMP') < 0:
        prep = "prep_HH_{}_HD1_{}_HD2_{}_OUT_{}_PD_{}_AT_{}".format(x[2],x[4], x[6], x[8], x[10], x[12])
    else:
        #prep = "prep_HH_{}_HD1_{}_HD2_{}_OUT_{}_PD_{}_AT_{}".format(x[2],x[4], x[6], x[8], x[10], x[12])
        prep = "prep_HHTMP_{}_HD1_{}_HD2_{}_OUT_{}_PD_{}_AT_{}".format(x[2],x[4], x[6], x[8], x[10], x[12])
        
    ready.append(prep)

ready.sort(key=lambda x:int(x.split('_')[8]))

rs = [(int(x.split('_')[8]), x) for x in ready]


r_start=0
dic = {}
r_end_last = 0
for r_end, prep in rs:
    if r_end > r_end_last:
        r_start = r_end_last
        r_end_last = r_end
        key = (r_start, r_end)
        if key not in dic:
            dic[key] = []
    dic[key].append(prep)

ensemble_preps = [[key, val] for key, val in dic.items()]
ensemble_preps.sort()

tm = datetime.datetime.now().strftime('%Y_%m_%d_%H')
with open('prepare_ensemble_origin.py') as f:
    text = f.read()
text = text.replace('FFFF', tm).replace('ENSEMBLE_LIST', str(ensemble_preps))
with open('prepare.py', 'w') as f:
    f.write(text)
    
if sys.argv[-1] == 'True':
    os.system("python3 pack.py")
else:
    for prep in ready:
        print(prep)
