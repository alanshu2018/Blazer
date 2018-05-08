
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
    name = "stack3"
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
         'ip','app','device','os','channel','day','hour','next_click',
         'ip_d_os_c_app','ip_c_os','ip_cu_c','ip_da_cu_h',
         'ip_cu_app','ip_app_cu_os','ip_cu_d','app_cu_chl','ip_d_os_cu_app',
         'ip_da_co','ip_app_co','ip_app_os_co','ip_d_co','app_chl_co','ip_ch_co',
         'ip_app_chl_co','app_d_co','app_os_co','ip_os_co','ip_d_os_co',
         'ip_app_h_co','ip_app_os_h_co','ip_d_h_co','app_chl_h_co',
         'ip_ch_h_co','ip_app_chl_h_co','app_d_h_co','app_os_h_co',
         'ip_os_h_co','ip_d_os_h_co','ip_da_chl_var_h','ip_chl_var_h',
         'ip_app_os_var_h','ip_app_chl_var_da','ip_app_chl_var_h',
         'app_os_var_da','app_d_var_h','app_chl_var_h','ip_app_chl_mean_h',
         'ip_chl_mean_h','ip_app_os_mean_h','ip_app_mean_h',
         'app_os_mean_h','app_mean_var_h','app_chl_mean_h',
         'next_click_shift',
         ],
        #['is_attributed'],
        #['is_attributed'],
        #['is_attributed'],
    ]
    train_predictors = [
        [
         'weight','click_id',
         'ip','app','device','os','channel','day','hour','next_click',
         'ip_d_os_c_app','ip_c_os','ip_cu_c','ip_da_cu_h',
         'ip_cu_app','ip_app_cu_os','ip_cu_d','app_cu_chl','ip_d_os_cu_app',
         'ip_da_co','ip_app_co','ip_app_os_co','ip_d_co','app_chl_co','ip_ch_co',
         'ip_app_chl_co','app_d_co','app_os_co','ip_os_co','ip_d_os_co',
         'ip_app_h_co','ip_app_os_h_co','ip_d_h_co','app_chl_h_co',
         'ip_ch_h_co','ip_app_chl_h_co','app_d_h_co','app_os_h_co',
         'ip_os_h_co','ip_d_os_h_co','ip_da_chl_var_h','ip_chl_var_h',
         'ip_app_os_var_h','ip_app_chl_var_da','ip_app_chl_var_h',
         'app_os_var_da','app_d_var_h','app_chl_var_h','ip_app_chl_mean_h',
         'ip_chl_mean_h','ip_app_os_mean_h','ip_app_mean_h',
         'app_os_mean_h','app_mean_var_h','app_chl_mean_h',
         'next_click_shift',
         ],
        #['predicted'],
        #['predicted'],
        #['predicted'],
    ]
    #Target in data
    target_file = "../newdata/train_v22.csv"
    target = 'click_id'
    weight = 'weight'

    skiprows = range(1,44903891)
    numrows = 40000000

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

def select_feature(conf,learner_name,exp_name):
    task_mode = "stacking"
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
    task_mode = "stacking"
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

exp_name="select_feature"
learner_name = "reg_etr"
learner_name = "clf_xgb_tree"
learner_name = "reg_keras_dnn"
learner_name = "reg_my_keras"
learner_name = "reg_lgb_native"
conf = Stack1Conf
logname = exp_name

select_feature(conf,learner_name,exp_name)
#main(conf, learner_name, exp_name)

"""
params_dict = {
'batch_norm': 'before_act',
'batch_size': 20000,
'hidden_activation': 'relu',
'hidden_dropout': 0.0,
'hidden_layers': 2,
'hidden_units': 64,
'input_dropout': 0.0,
'nb_epoch': 1, #20
'optimizer': 'rmsprop',
}

eval(params_dict, conf, learner_name, exp_name)
"""


