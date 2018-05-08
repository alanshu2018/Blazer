
import os
import sys
import time
import gc
from optparse import OptionParser

from lightgbm import LGBMClassifier, LGBMRegressor

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from sklearn.model_selection import *

import config
from utils import dist_utils, logging_utils, pkl_utils, time_utils
#from utils.xgb_utils import XGBRegressor, HomedepotXGBClassifier as XGBClassifier
from utils.rgf_utils import RGFRegressor
from utils.skl_utils import SVR, LinearSVR, KNNRegressor, AdaBoostRegressor, RandomRidge
try:
    from utils.keras_utils import KerasDNNRegressor
except:
    pass
from model_param_space import ModelParamSpace

from Tunner import *

exp_name = "exp31"
desc = """
Tune with data v31 and train
"""

target = 'click_id'
categorical = ['ip','app', 'device', 'os', 'channel', 'hour']

predictors=[
    'ip','app','device','channel','os','hour',
    'ip_d_os_c_app','ip_cu_c','ip_da_cu_h','ip_cu_app',
    'ip_app_cu_os','ip_cu_d','ip_d_os_cu_app','ip_da_co','ip_app_co','ip_d_co','app_chl_co','ip_ch_co',
    'app_d_co','ip_os_co','ip_d_os_co','ip_app_h_co','ip_app_os_h_co','ip_d_h_co','app_chl_h_co','ip_ch_h_co',
    'app_os_h_co','ip_os_h_co','ip_d_os_h_co','ip_da_chl_var_h','ip_chl_var_h','ip_app_os_var_h',
    'ip_app_chl_var_da','ip_app_chl_var_h','app_chl_var_h','ip_app_os_mean_h','app_os_mean_h',
    'ip_channel_prevClick','ip_app_device_os_channel_prevClick','ip_os_device_app_prevClick',
    'ip_app_device_os_channel_nextClick','ip_os_device_nextClick',
    'ip_os_device_app_nextClick',
    ]

out_dir = "../newdata/Output/"
sub_dir = "../newdata/Output/Subm/"
class Stack1Conf(object):
    name = exp_name
    test = [
        "../newdata/test_v31.csv",

    ]
    train = [
        "../newdata/train_v31.csv",
    ]
    test_predictors = [
        [
            'click_id',
            'ip','app','device','channel','os','hour',
            'ip_d_os_c_app','ip_cu_c','ip_da_cu_h','ip_cu_app',
            'ip_app_cu_os','ip_cu_d','ip_d_os_cu_app','ip_da_co','ip_app_co','ip_d_co','app_chl_co','ip_ch_co',
            'app_d_co','ip_os_co','ip_d_os_co','ip_app_h_co','ip_app_os_h_co','ip_d_h_co','app_chl_h_co','ip_ch_h_co',
            'app_os_h_co','ip_os_h_co','ip_d_os_h_co','ip_da_chl_var_h','ip_chl_var_h','ip_app_os_var_h',
            'ip_app_chl_var_da','ip_app_chl_var_h','app_chl_var_h','ip_app_os_mean_h','app_os_mean_h',
            'ip_channel_prevClick','ip_app_device_os_channel_prevClick','ip_os_device_app_prevClick',
            'ip_app_device_os_channel_nextClick','ip_os_device_nextClick',
            'ip_os_device_app_nextClick',
        ],
    ]
    train_predictors = [
        [
            'click_id',
            'ip','app','device','channel','os','hour',
            'ip_d_os_c_app','ip_cu_c','ip_da_cu_h','ip_cu_app',
            'ip_app_cu_os','ip_cu_d','ip_d_os_cu_app','ip_da_co','ip_app_co','ip_d_co','app_chl_co','ip_ch_co',
            'app_d_co','ip_os_co','ip_d_os_co','ip_app_h_co','ip_app_os_h_co','ip_d_h_co','app_chl_h_co','ip_ch_h_co',
            'app_os_h_co','ip_os_h_co','ip_d_os_h_co','ip_da_chl_var_h','ip_chl_var_h','ip_app_os_var_h',
            'ip_app_chl_var_da','ip_app_chl_var_h','app_chl_var_h','ip_app_os_mean_h','app_os_mean_h',
            'ip_channel_prevClick','ip_app_device_os_channel_prevClick','ip_os_device_app_prevClick',
            'ip_app_device_os_channel_nextClick','ip_os_device_nextClick',
            'ip_os_device_app_nextClick',
        ],
    ]
    # Number of dimensions to combine
    ndim = 2
    #Target in data
    target_file = "../newdata/train_v31.csv"
    target = 'click_id'
    weight = 'weight'

def main(conf,learner_name,exp_name):
    task_mode = "stacking"
    feature_name = conf.name
    max_evals = 10
    refit_once = True
    logname = "%s_[Feat@%s]_[Learner@%s]_hyperopt_%s.log"%(
        exp_name, feature_name, learner_name, time_utils._timestamp())


    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    optimizer = TaskOptimizer(task_mode, learner_name,
                              conf, logger, max_evals, verbose=True,
                              refit_once=refit_once, plot_importance=False)
    optimizer.run()

def eval(params_dict, conf,learner_name,exp_name):
    task_mode = "stacking"
    feature_name = conf.name
    max_evals = 10
    refit_once = True
    logname = "%s_[Feat@%s]_[Learner@%s]_hyperopt_%s.log"%(
        exp_name, feature_name, learner_name, time_utils._timestamp())


    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    optimizer = TaskOptimizer(task_mode, learner_name,
                              conf, logger, max_evals, verbose=True,
                              refit_once=refit_once, plot_importance=False)
    optimizer.rerun(params_dict)

learner_name = "reg_etr"
learner_name = "clf_xgb_tree"
learner_name = "reg_keras_dnn"
learner_name = "reg_lgb_native"
#learner_name = "reg_my_keras"
#learner_name = "reg_rgf"

print("****>> Use learner:{}".format(learner_name))

conf = Stack1Conf
logname = exp_name

main(conf, learner_name, exp_name)
"""
params_dict={
	'device': 'gpu', 
	'subsample_freq': 1, 
	'reg_lambda': 0, 
	'min_split_gain': 0, 
	'min_child_samples': 100, 
	'reg_alpha': 0, 
	'learning_rate': 0.02, 
	'objective': 'binary', 
	'scale_pos_weight': 99.7, 
	'verbose': 1, 
	'max_depth': 10, 
	'subsample_for_bin': 200000, 
	'metric': 'auc', 
	'subsample': 0.7, 
	'min_child_weight': 0.06184481608893035, 
	'num_leaves': 36, 
	'boosting_type': 'gbdt', 
	'nthread': 8, 
	'colsample_bytree': 0.9
}
params_dict={'batch_size': 18000, 'hidden_activation': 'elu', 'nb_epoch': 9, 'hidden_dropout': 0.30000000000000004, 'hidden_units': 32, 'hidden_layers': 2, 'optimizer': 'rmsprop', 'input_dropout': 0.15000000000000002, 'batch_norm': 'no'}
eval(params_dict, conf, learner_name, exp_name)
"""


