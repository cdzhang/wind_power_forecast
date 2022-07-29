import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def KDD_Score(pred, true, raw_data_ls, s_ind=0, num_turb=134, stride=1):
    # pred, true: prediction and label of all turbines
    # raw_data_ls: raw data list, each turbine has a dataframe
    # s_ind: the index of starting the evaluation
    # pred shape: (N, pred_len, 1)
    # preds shape: (N//num_turb, num_turb, pred_len, 1)
    # 
    conds = []
    # for raw_data in raw_data_ls:
    #     cond = (raw_data['Patv'] <= 0) & (raw_data['Wspd'] > 2.5) | \
    #            (raw_data['Pab1'] > 89) | (raw_data['Pab2'] > 89) | (raw_data['Pab3'] > 89) | \
    #            (raw_data['Wdir'] < -180) | (raw_data['Wdir'] > 180) | (raw_data['Ndir'] < -720) | (raw_data['Ndir'] > 720)
    #     conds.append(cond)
    # here cond can use abnormal col, without re-define    
    for raw_data in raw_data_ls:
        conds.append(raw_data['abnormal'])
    # 173798(id*t), 288, 1
    num_samples, out_seq_len, out_seq_dim = pred.shape
    assert num_samples % num_turb == 0 # 1297
    size = num_samples // num_turb
    
    # 1297 = 1584 - 288 + 1
    assert size == raw_data_ls[0].shape[0] - out_seq_len + 1

    all_mae, all_rmse = [], []
    for i in range(0, size, stride):
        maes, rmses = [], []
        for j in range(num_turb):
            # compute mae and rmse of each turbine
            indices = np.where(~conds[j][i:out_seq_len + i])
            prediction = pred[j*size+i]
            prediction = prediction[indices]
            targets = true[j*size+i]
            targets = targets[indices]
            rmse = RMSE(prediction[s_ind:] / 1000, targets[s_ind:] / 1000)
            mae = MAE(prediction[s_ind:] / 1000, targets[s_ind:] / 1000)
            if mae != mae or rmse != rmse:
                continue
            maes.append(mae)
            rmses.append(rmse)
        
        all_mae.append(np.array(maes).sum())
        all_rmse.append(np.array(rmses).sum())
    
    avg_mae = np.array(all_mae).mean()
    avg_rmse = np.array(all_rmse).mean()
    total_score = (avg_mae + avg_rmse) / 2

    print('\n --- Final MAE: {}, RMSE: {} ---'.format(avg_mae, avg_rmse))
    print('--- Final Score --- \n\t{}'.format(total_score))
    return avg_mae, avg_rmse, total_score

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe



def cal_kdd_metrics(preds, trues, masks):
    """
    support batch calculation
    preds.shape = true.shape = masks.shape = [batch_size, 288, 134] 或 [288, 134]
    preds: prediction (need to inverse if it has been normalized)
    trues: true label
    mask: masks_true
    """
    assert preds.shape[-1] == 134
    assert preds.shape[-2] == 288
    assert preds.shape == trues.shape == masks.shape
    
    preds = preds / 1000 # change unit
    trues = trues / 1000 # change unit
    
    mse = ((trues-preds)**2)*masks 
    mse = np.sqrt(mse.sum(axis=-2)/288)  #  for each turbine, sum along the time dimension. shape: [batch_size, 134] or [134]
    
    mae = np.abs(trues-preds)*masks
    mae = mae.sum(axis=-2)/288           #  for each turbine, sum along the time dimension. shape: [batch_size, 134] or [134]
    
    s = 0.5*mse + 0.5*mae
    S = s.sum(axis=-1)
    return np.mean(S), np.mean(mse), np.mean(mae)

