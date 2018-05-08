
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

class Stack1Conf(object):
    name = "stack4"
    test = [
        "../newdata/test_v22.csv",
        #'../sub_v20_20180419002702_lgb_lr_0.2_num_leaves_7_max_depth_3.csv',
        #'../sub_wordbatch_fm_ftrl_16_20180417231311.csv',
        #'../sub_wordbatch_fm_ftrl_wordbatch_fm_ftrl_16.csv',
    ]
    train = [
        "../newdata/train_v22.csv",
        #'../cv_v20_20180419002702_lgb_lr_0.2_num_leaves_7_max_depth_3.csv',
        #'../cv_wordbatch_fm_ftrl_16_20180417231311.csv',
        #'../cv_wordbatch_fm_ftrl_wordbatch_fm_ftrl_16.csv',
    ]
    test_predictors = [
        [
         'weight','click_id',
         'app_cu_chl', 'ip', 'app', 'ip_cu_c', 'ip_da_chl_var_h', 'ip_app_os_var_h', 'device', 'app_d_h_co',
         'app_chl_var_h', 'ip_d_os_c_app', 'app_chl_h_co', 'next_click', 'app_d_co', 'ip_d_co', 'app_os_var_da',
         'ip_da_cu_h', 'channel', 'app_os_h_co', 'app_chl_co', 'ip_app_chl_var_h', 'ip_os_co', 'ip_app_co',
         'app_os_co', 'ip_d_os_cu_app', 'app_d_var_h', 'ip_chl_var_h', 'hour', 'ip_app_cu_os', 'ip_d_h_co',
         'ip_app_chl_var_da', 'app_os_mean_h', 'ip_cu_app', 'ip_os_h_co', 'os','next_click_shift'
         ],
        #['is_attributed'],
        #['is_attributed'],
        #['is_attributed'],
    ]
    train_predictors = [
        [
        'weight','click_id',
        'app_cu_chl', 'ip', 'app', 'ip_cu_c', 'ip_da_chl_var_h', 'ip_app_os_var_h', 'device', 'app_d_h_co',
        'app_chl_var_h', 'ip_d_os_c_app', 'app_chl_h_co', 'next_click', 'app_d_co', 'ip_d_co', 'app_os_var_da',
        'ip_da_cu_h', 'channel', 'app_os_h_co', 'app_chl_co', 'ip_app_chl_var_h', 'ip_os_co', 'ip_app_co',
        'app_os_co', 'ip_d_os_cu_app', 'app_d_var_h', 'ip_chl_var_h', 'hour', 'ip_app_cu_os', 'ip_d_h_co',
        'ip_app_chl_var_da', 'app_os_mean_h', 'ip_cu_app', 'ip_os_h_co', 'os','next_click_shift'
         ],
        #['predicted'],
        #['predicted'],
        #['predicted'],
    ]
    #Target in data
    target_file = "../newdata/train_v22.csv"
    target = 'click_id'
    weight = 'weight'

    # Keep Last 40000000
    skiprows = range(1,24903889)
    numrows = 40000000

expname = "stack4"
def main(conf,learner_name,exp_name):
    task_mode = expname
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

def select_feature(conf,learner_name,exp_name):
    task_mode = expname
    feature_name = conf.name
    max_evals = 20
    refit_once = True
    logname = "%s_[Feat@%s]_[Learner@%s]_hyperopt_%s.log"%(
        exp_name, feature_name, learner_name, time_utils._timestamp())


    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    optimizer = TaskOptimizer(task_mode, learner_name,
                              conf, logger, max_evals, verbose=True,
                              refit_once=refit_once, plot_importance=False)
    given_predictors =['0_ip','0_app','0_device','0_os','0_channel','0_day','0_hour','0_next_click','0_next_click_shift']
    optimizer.select_features(given_predictors)

def eval(params_dict, conf,learner_name,exp_name):
    task_mode = expname
    feature_name = conf.name
    max_evals = 20
    refit_once = True
    logname = "%s_[Feat@%s]_[Learner@%s]_hyperopt_%s.log"%(
        exp_name, feature_name, learner_name, time_utils._timestamp())


    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    optimizer = TaskOptimizer(task_mode, learner_name,
                              conf, logger, max_evals, verbose=True,
                              refit_once=refit_once, plot_importance=False)
    optimizer.rerun(params_dict)

exp_name=expname #select_feature"
learner_name = "reg_etr"
learner_name = "clf_xgb_tree"
learner_name = "reg_keras_dnn"
learner_name = "reg_my_keras"
learner_name = "reg_lgb_native"
conf = Stack1Conf
logname = exp_name

#select_feature(conf,learner_name,exp_name)
#main(conf, learner_name, exp_name)

params_dict = {
    'boosting_type': 'gbdt',
    'colsample_bytree': 0.9,
    'learning_rate': 0.106,
    'max_depth': 3,
    'metric': 'auc',
    'min_child_samples': 100,
    'min_child_weight': 3.72113038838e-06,
    'min_split_gain': 0,
    'nthread': 8,
    'num_leaves': 8,
    'objective': 'binary',
    'reg_alpha': 0,
    'reg_lambda': 0,
    'scale_pos_weight': 400,
    'subsample': 0.7,
    'subsample_for_bin': 200000,
    'subsample_freq': 1,
    'verbose': 1,
}

eval(params_dict, conf, learner_name, exp_name)


