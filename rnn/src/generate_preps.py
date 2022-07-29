import os

with open('prepare_origin.py') as f:
    origin_prep = f.read()

dic = {
    'HH': [2, 4, 7],
    'HID_SZ1': [64, 128],
    'HID_SZ2': [8, 16],
    'OUT': [12, 36, 72, 144, 288],
    'PARTIAL_DATA': ['None', 140],
    'ATTN': ['True', 'False']
}
for key in dic.keys():
    dic[key] = [str(x) if type(x)==int else x for x in dic[key]]
print(dic)
for HH in dic['HH']:
    for HID_SZ1 in dic['HID_SZ1']:
        for HID_SZ2 in dic['HID_SZ2']:
            for OUT in dic['OUT']:
                for PARTIAL_DATA in dic['PARTIAL_DATA']:
                    for ATTN in dic['ATTN']:
                        surfix = 'HH_{}_HD1_{}_HD2_{}_OUT_{}_PD_{}_AT_{}'.format(HH, HID_SZ1, HID_SZ2, OUT, PARTIAL_DATA, ATTN)
                        text = origin_prep.replace('HH', HH).replace('HIDSZ1', HID_SZ1).replace('HIDSZ2', 
                        HID_SZ2).replace('OUT', OUT).replace('PARTIAL_DATA', PARTIAL_DATA).replace('ATTN', ATTN)
                        prep_file = "prep_{}.py".format(surfix)
                        run_file = 'run_{}.sh'.format(surfix)
                        submit_file = 'falcon_submit_{}.sh'.format(surfix)
                        log_file = 'logs/log_{}.txt'.format(surfix)
                        with open(prep_file, 'w') as f:
                            f.write(text)
                        with open(run_file, 'w') as f:
                            f.write("python3 pipeline.py {} > {} 2>&1".format(prep_file, log_file))
                    
