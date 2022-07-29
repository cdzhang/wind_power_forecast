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


def cal_kdd_metrics_v2(preds, trues, masks):
    """
    
    support batch data
    preds.shape = true.shape = masks.shape = [batch_size, output_len, 134] or [output_len, 134]
    preds: predictions
    trues: true values
    mask: masks of true values
    """
    assert preds.shape[-1] == 134
    #assert preds.shape[-2] == 288
    #assert preds.shape == trues.shape == masks.shape
    
    mask_sum = masks.sum(axis=-2) + 0.0001
    preds = preds / 1000 # change unit (kW->MW)
    trues = trues / 1000 
    
    mse = ((trues-preds)**2)*masks   # 
    mse = np.sqrt(mse.sum(axis=-2)/mask_sum)  
    
    mae = np.abs(trues-preds)*masks  
    mae = mae.sum(axis=-2)/mask_sum      
    s = 0.5*mse + 0.5*mae 
    S = s.sum(axis=-1)   
    return np.mean(S), np.mean(mse.sum(axis=-1)), np.mean(mae.sum(axis=-1))
