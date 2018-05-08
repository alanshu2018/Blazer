
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

exp_name = "stack16_1"
out_dir = "../newdata/Output/"
sub_dir = "../newdata/Output/Subm/"
class Stack1Conf(object):
    name = exp_name
    test = [
        "../newdata/test_basic.csv",
        sub_dir + 'sub_v20_20180419002702_lgb_lr_0.2_num_leaves_7_max_depth_3.csv',
        sub_dir + 'sub_wordbatch_fm_ftrl_16_20180417231311.csv',
        sub_dir + 'sub_wordbatch_fm_ftrl_wordbatch_fm_ftrl_16.csv',
        sub_dir + 'sub_pred.Feat@stack10_Learner@reg_lgb_native.[Mean0.022502]_[Std0.003067].csv',
        sub_dir + 'sub_pred.Feat@stack13_Learner@reg_lgb_native.[Mean0.017455]_[Std0.002860].csv',
        sub_dir + 'sub_pred.Feat@stack1_Learner@reg_lgb_native.[Mean0.018308]_[Std0.003611].csv',
        sub_dir + 'sub_pred.Feat@stack1_Learner@reg_my_keras.[Mean0.017438]_[Std0.002688].csv',
        sub_dir + 'sub_pred.Feat@stack4_Learner@reg_lgb_native.[Mean0.000000]_[Std0.000000].csv',
        sub_dir + 'sub_pred.Feat@stack8_Learner@reg_lgb_native.[Mean0.018597]_[Std0.003297].csv',
        sub_dir + 'sub_pred_fmftrl_22.csv',
        sub_dir + 'sub_pred_fmftrl_26_20180503172030.csv',
    ]
    train = [
        "../newdata/train_basic.csv",
        out_dir + 'cv_v20_20180419002702_lgb_lr_0.2_num_leaves_7_max_depth_3.csv',
        out_dir + 'cv_wordbatch_fm_ftrl_16_20180417231311.csv',
        out_dir + 'cv_wordbatch_fm_ftrl_wordbatch_fm_ftrl_16.csv',
        out_dir + 'cv_pred.Feat@stack10_Learner@reg_lgb_native.csv',
        out_dir + 'cv_pred.Feat@stack13_Learner@reg_lgb_native.csv',
        out_dir + 'cv_pred.Feat@stack1_Learner@reg_lgb_native.csv',
        out_dir + 'cv_pred.Feat@stack1_Learner@reg_my_keras.csv',
        out_dir + 'cv_pred.Feat@stack4_Learner@reg_lgb_native.csv',
        out_dir + 'cv_pred.Feat@stack8_Learner@reg_lgb_native.csv',
        out_dir + 'cv_pred_fmftrl_22.csv',
        out_dir + 'cv_pred_fmftrl_26.csv',
    ]
    test_predictors = [
        [
            'weight','click_id',
            #'ip','app','device','channel','os','hour',
        ],
        ['is_attributed'],
        ['is_attributed'],
        ['is_attributed'],
        ['is_attributed'],
        ['is_attributed'],
        ['is_attributed'],
        ['is_attributed'],
        ['is_attributed'],
        ['is_attributed'],
        ['is_attributed'],
        ['is_attributed'],
    ]
    train_predictors = [
        [
            'weight','click_id',
            #'ip','app','device','channel','os','hour',
        ],
        ['predicted'],
        ['predicted'],
        ['predicted'],
        ['predicted'],
        ['predicted'],
        ['predicted'],
        ['predicted'],
        ['predicted'],
        ['predicted'],
        ['predicted'],
        ['predicted'],
    ]
    # Number of dimensions to combine
    ndim = 11
    #Target in data
    target_file = "../newdata/train_v22.csv"
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
learner_name = "reg_my_keras"
#learner_name = "reg_rgf"

print("****>> Use learner:{}".format(learner_name))

conf = Stack1Conf
logname = exp_name

main(conf, learner_name, exp_name)

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


