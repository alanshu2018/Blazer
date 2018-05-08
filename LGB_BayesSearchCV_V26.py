#coding: utf-8

import pandas as pd
import numpy as np
import lightgbm as lgb

import numpy as np
#import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

# SETTINGS - CHANGE THESE TO GET SOMETHING MEANINGFUL
ITERATIONS = 100 # 1000
TRAINING_SIZE = 2000000 # 20000000
TEST_SIZE = 25000

dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'float16',
    'click_id'      : 'uint32',
    'next_click':   'uint8',
    'next_click_shift':'float16',
    'X0':'uint16',
    'X1':'uint16',
    'X2':'uint16',
    'X3':'uint16',
    'X4':'uint16',
    'X5':'uint16',
    'X6':'uint16',
    'X7':'uint16',
    'X8':'uint16',
    'XX0':'uint16',
    'XX1':'uint16',
    'XX2':'uint16',
    'XX3':'uint16',
    'XX4':'uint16',
    'XX5':'uint16',
    'XX6':'uint16',
    'XX7':'uint16',
    'XX8':'uint16',
    'ip_tchan_var':'float16',
    'ip_app_os_var':'float16',
    'ip_app_channel_var_day':'float16',
    'ip_app_channel_mean_hour':'float16',
    'weight':'float16',
    'ip_tcount':'float16',
    'ip_tchan_count':'float16',
    'ip_app_count':'float16',
    'ip_app_os_count':'float16',
    'ip_app_os_var':'float16',
    'ip_app_channel_var_day':'float16',
    'ip_app_channel_mean_hour':'float16',
    'ip_d_os_c_app':'uint32',
    'ip_c_os':'uint32',
    'ip_cu_c':'uint16',
    'ip_da_cu_h':'uint16',
    'ip_cu_app':'uint16',
    'ip_app_cu_os':'uint16',
    'ip_cu_d':'uint16',
    'app_cu_chl':'uint16',
    'ip_d_os_cu_app':'uint16',
    'ip_da_co':'uint16',
    'ip_app_co':'uint16',
    'ip_app_os_co':'uint16',
    'ip_d_co':'uint16',
    'app_chl_co':'uint16',
    'ip_ch_co':'uint16',
    'ip_app_chl_co':'uint16',
    'app_d_co':'uint16',
    'app_os_co':'uint16',
    'ip_os_co':'uint16',
    'ip_d_os_co':'uint16',
    'ip_app_h_co':'uint16',
    'ip_app_os_h_co':'uint16',
    'ip_d_h_co':'uint16',
    'app_chl_h_co':'uint16',
    'ip_ch_h_co':'uint16',
    'ip_app_chl_h_co':'uint16',
    'app_d_h_co':'uint16',
    'app_os_h_co':'uint16',
    'ip_os_h_co':'uint16',
    'ip_d_os_h_co':'uint16',
    'ip_da_chl_var_h':'float16',
    'ip_chl_var_h':'float16',
    'ip_app_os_var_h':'float16',
    'ip_app_chl_var_da':'float16',
    'ip_app_chl_var_h':'float16',
    'app_os_var_da':'float16',
    'app_d_var_h':'float16',
    'app_chl_var_h':'float16',
    'ip_app_chl_mean_h':'float16',
    'ip_chl_mean_h':'float16',
    'ip_app_os_mean_h':'float16',
    'ip_app_mean_h':'float16',
    'app_os_mean_h':'float16',
    'app_mean_var_h':'float16',
    'app_chl_mean_h':'float16',
    'ip_channel_prevClick':'float16',
    'ip_os_prevClick':'float16',
    'ip_app_device_os_channel_nextClick':'float16',
    'ip_os_device_nextClick':'float16',
    'ip_os_device_app_nextClick':'float16',
}

predictors = [
'app_cu_chl', 'ip', 'app', 'ip_cu_c', 'ip_da_chl_var_h', 'ip_app_os_var_h', 'device', 'app_d_h_co', #8
'app_chl_var_h', 'ip_d_os_c_app', 'app_chl_h_co', 'app_d_co', 'ip_d_co', 'app_os_var_da', #7
'ip_da_cu_h', 'channel', 'app_os_h_co', 'app_chl_co', 'ip_app_chl_var_h', 'ip_os_co', 'ip_app_co', #7
'app_os_co', 'ip_d_os_cu_app', 'app_d_var_h', 'ip_chl_var_h', 'hour', 'ip_app_cu_os', 'ip_d_h_co', #7
'ip_app_chl_var_da', 'app_os_mean_h', 'ip_cu_app', 'ip_os_h_co', 'os', #6
'ip_channel_prevClick','ip_os_prevClick','ip_app_device_os_channel_nextClick',
'ip_os_device_nextClick','ip_os_device_app_nextClick',
#'next_click','next_click_shift',
]

# Load data
X = pd.read_csv(
    '../newdata/train_v26_8.csv',
    skiprows=range(1,20000000-TRAINING_SIZE),
    nrows=TRAINING_SIZE,
    usecols=predictors + ['click_id','weight'],
    dtype = dtypes,
)

# Split into X and y
y = X['click_id'].values
W = X['weight'].values
X = X[predictors].values

#X = X.drop(['click_time','is_attributed', 'attributed_time'], axis=1)
import gc
D = 2 ** 21
batchsize = 10000000
def df2csr(data):
    data_shape = data.shape

    print("data_shape={}".format(data_shape))
    str_array = np.apply_along_axis(lambda row: " ".join(["{}{}".format(chr(65+i),row[i]) for i in range(data_shape[1])]), 1, data)
    del(data)
    gc.collect()
    print("str_array_shape={}".format(str_array.shape))
    return str_array

"""
import wordbatch
from wordbatch.extractors import WordHash
from wordbatch.models import FM_FTRL
from wordbatch.data_utils import *

wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
                                                         "lowercase": False, "n_features": D,
                                                         "norm": None, "binary": True})
                             , minibatch_size=batchsize // 80, procs=8, freeze=True, timeout=1800, verbose=0)

X = df2csr(X.values)
X = wb.transform(X)
gc.collect()
"""

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

    # Get current parameters and the best parameters
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))

    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")

# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = lgb.LGBMRegressor(
        objective='binary',
        metric='auc',
        n_jobs=1,
        verbose=0
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (1, 100),
        'max_depth': (1, 256),
        'min_child_samples': (0, 50),
        'max_bin': (1000, 10000),
        'subsample': (0.01, 1.0, 'uniform'),
        'subsample_freq': (0, 10),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'min_child_weight': (0, 10),
        'subsample_for_bin': (100000, 500000),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        'n_estimators': (50, 100),
    },
    scoring = 'roc_auc',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = ITERATIONS,
    verbose = 0,
    refit = True,
    random_state = 42
)

# Fit the model
result = bayes_cv_tuner.fit(X, y, callback=status_print)