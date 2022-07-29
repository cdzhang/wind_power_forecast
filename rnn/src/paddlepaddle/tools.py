import os
import numpy as np
import paddle
from rnn.src.metrics import cal_kdd_metrics_v2


def adjust_learning_rate(optimizer, epoch, args):
    # type: (paddle.optimizer.Adam, int, dict) -> None
    """
    Desc:
        Adjust learning rate
    Args:
        optimizer:
        epoch:
        args:
    Returns:
        None
    """
    # lr = args.lr * (0.2 ** (epoch // 2))
    lr_adjust = {}
    if args["lr_adjust"] == 'type1':
        # learning_rate = 0.5^{epoch-1}
        lr_adjust = {epoch: args["lr"] * (0.50 ** (epoch - 1))}
    elif args["lr_adjust"] == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        optimizer.set_lr(lr)

class EarlyStopping(object):
    """
    Desc:
        EarlyStopping
    """
    def __init__(self, args, delta=0):
        self.patience = args['patience']
        self.verbose = args['verbose']
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_model = False
        self.delta = delta
    
    def save_checkpoint(self, val_loss, model, model_file):
        # type: (nn.MSELoss, BaselineGruModel, str, int) -> None
        """
        Desc:
            Save current checkpoint
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        path = os.path.dirname(model_file)
        if not os.path.isdir(path):
            os.mkdir(path)
        
        self.best_model = True
        self.val_loss_min = val_loss
        if self.verbose:
            print("better model found, saving in ", model_file)
        paddle.save(model.state_dict(), model_file)

    def __call__(self, val_loss, model, model_file, init=False):
        # type: (nn.MSELoss, BaselineGruModel, str, int) -> None
        """
        Desc:
            __call__
        Args:
            val_loss: the validation loss
            model: the model
            path: the path to be saved
            tid: turbine ID
        Returns:
            None
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if not init:
                self.save_checkpoint(val_loss, model, model_file)
        
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.best_model = False
            if self.counter >= self.patience:
                if self.verbose:
                    print("early stopping criteria reached, best score is", -self.best_score)
                self.early_stop = True
        
        else:
            self.best_score = score
            self.update_hidden = True
            self.save_checkpoint(val_loss, model, model_file)
            self.counter = 0


def val(model, val_loader, args):
    """

    """
    #loss_fun = paddle.nn.MSELoss(reduction='mean')
    model.eval()
    val_loss = []
    scaler = args['stored_args']['stored_scaler']
    with paddle.no_grad():
        for x_enc, x_dec, fture_y, value_y, mask_x, mask_y in val_loader:
            x_enc = x_enc.astype('float32')
            x_dec = x_dec.astype('float32')
            fture_y = fture_y.astype('float32')
            value_y = value_y.astype('float32').detach().numpy()[:,:,:134]


            mask_y = mask_y.astype('float32').detach().numpy()[:,:,:134]

            y_pred = model(x_enc, feature_dec=fture_y).detach().numpy()[:,:,:134]
            loss = ((y_pred-value_y)**2*mask_y).sum()/(mask_y.sum()+0.0001)
            val_loss.append(loss)
    return np.mean(val_loss)


def val_lstm_enc_f(model, val_loader, args, kdd_score=False):
    """

    """
    loss_fun = paddle.nn.MSELoss(reduction='mean')
    model.eval()
    val_loss = []
    with paddle.no_grad():
        for x_enc, mask_enc, x2_enc, fture_dec, value_dec, fture2_dec, value2_dec, mask_dec in val_loader:
            
            if fture_dec.shape[-1] > 0:
                fture_dec = fture_dec.astype('float32')
            
            
            if fture2_dec.shape[-1] > 0:
                fture2_dec = fture2_dec.astype('float32')
            
            if args['paddlepaddle_model'] == 'LSTM_enc2_tp2':
                scaler = args['stored_args']['stored_scaler']
                u, std = scaler.mean, scaler.std
                y_pred, y_pred2 = model(u, std, x_enc, x2_enc, fture_dec, fture2_dec)
            else:
                y_pred, y_pred2 = model(x_enc, x2_enc, fture_dec, fture2_dec)
            
            
            
            loss = loss_fun(y_pred[:,:,:134]*mask_dec[:,:,:134], 
                            value_dec[:,:,:134]*mask_dec[:,:,:134])
            
            val_loss.append(loss.item())
            
            if kdd_score:
                mask_dec = mask_dec.detach().numpy()
                y_pred *= args['max_patv']
                value_dec *= args['max_patv']
                scaler = args['stored_args']['stored_scaler']
                y_pred = scaler.inverse_transform(y_pred).detach().numpy()
                value_dec = scaler.inverse_transform(value_dec).detach().numpy()
                kdd_score = cal_kdd_metrics_v2(y_pred[:,:,:134], value_dec[:,:,:134],
                                               mask_dec[:,:,:134])[-1]
    return np.mean(val_loss), kdd_score


def val_kdd_score(model, val_loader):
    """
    计算kdd score
    """
    val_loss = []
    for x_enc, x_dec, fture_y, value_y, mask_x, mask_y in val_loader:
        x_enc = x_enc.astype('float32')
        x_dec = x_dec.astype('float32')
        fture_y = fture_y.astype('float32')
        value_y = value_y.astype('float32')
        dec_size = x_dec.shape[1]

        y_pred = model(x_enc, feature_dec=fture_y)
        loss = criterion(y_pred, value_y)
        val_loss.append(loss.detach().numpy())
    return np.mean(val_loss)


def prior_test_metrics(model, test_loader, args):
    test_mse = []
    test_mae = []
    for x_enc, x_dec, fture_y, value_y in test_loader:
        x_enc = x_enc.astype('float32')
        x_dec = x_dec.astype('float32')
        fture_y = fture_y.astype('float32')
        value_y = value_y.astype('float32')
        dec_size = x_dec.shape[1]

        if not total:
            y_pred = model(x_enc, feature_dec=fture_y).detach().numpy()[:,:,:-1] * max_power 
            y_true = value_y.detach().numpy()[:,:,:-1] * max_power
            y_pred_total = y_pred.sum(axis=-1) / 1000
            y_true_total = y_true.sum(axis=-1) / 1000
        else:
            y_pred_total = model(x_enc, feature_dec=fture_y).detach().numpy()[:,:,-1] * max_power * max_total_power / 1000
            y_true_total = value_y.detach().numpy()[:,:,-1] * max_power * max_total_power / 1000

        mse = np.sqrt(((y_pred_total-y_true_total)**2).mean(axis=-1)).mean()
        mae = np.abs(y_pred_total-y_true_total).mean(axis=-1).mean()
        test_mse.append(mse)
        test_mae.append(mae)
    return np.mean(test_mse), np.mean(test_mae)

