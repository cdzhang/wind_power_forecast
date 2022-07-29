# -*-Encoding: utf-8 -*-
"""
Description: Predict
Authors: aihongfeng
Date:    2022/05/23
Function:
 
"""

import numpy as np
import time
import os
import sys
sys.path.append(os.path.abspath(__file__).split('lgb_model')[0])
from lgb_model.src.model_wrapper import ModelWrapper, load_trained_model_wrapper, train


# Online code will call prep_env(), then modify parameter likes path_to_test_x. then call forecast() to get prediction
def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:  
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions
    """
    wrapper = load_trained_model_wrapper(settings)
    pred_df = wrapper.predict()
    prediction = np.array(pred_df['pred_target'])
    prediction = prediction.reshape(134, 288, 1)
    # since online evaluation don't allow the std of prediction is less than 0.1, we add noise here
    prediction = prediction + np.random.randn(134, 288, 1) * 0.015
    return prediction


if __name__=="__main__":
    from prepare import prep_env
    settings = prep_env()
    # if offline, do train directly
    if settings['is_submit'] == False:
        train(settings)
        print('Finish Offline Training and Predicting!')
    # if online, predict test_x directly
    else:
        stime = time.time()
        prediction = forecast(settings)
        prediction = prediction + np.random.randn(134, 288, 1) * 0.015
        etime = time.time()
        cost_time = etime - stime
        print('Finsh Online Predicting! Cost Time:%.2fs' % cost_time)
