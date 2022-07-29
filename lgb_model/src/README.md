# Parameter Setting
If you want offline training, go `prepare.py`, set "is_submit"=False. Otherwise set False to do online inference. 

# Offline Training and Evaluating
```
# Data Preparing
python wind_turbine_data.py
# Model Training and Evaluating
python predict.py
```
Note: Since we use tsfresh to auto-generating features, it will takes around 8-9 hours under the CPU with 7 cores and 70G memory.

# Online Infering
Online platform will unzip the file to make online inference.
```
# zip lgb_model/src
zip -p -r src.zip src -x src_lgb/wtbdata_245days_preprocess.csv
```

# Script Description
`prepare.py`：Parameter Setting. It has `prep_env()` providing parameter dictionary.
`wind_turbine_data.py`: Data Preparing. this script contains the following implementationss: abnormal preprocessing, feature engineerging, label generating, data spliting, standardlizaiton.
`predict.py`: Call for the Prediction of the given test dataset.
`model_wrapper.py`: Model Training, Evaluating and Prediction.
`metrics.py`: KDD evaluation metrics for the offline evaluation.
`utils.py`: common utilities function.

The middle file while running the code：
`scaler.npy`: it contains mu and std of each feature of each turbine。
`wtbdata_245days_preprocess.csv`: the preprocessed dataset after running `wind_turbine_data.py`, which can be used for training. It is avoid of having to repeatedly prepare the dataset every time the model is iteratively tuned
