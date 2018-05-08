
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

exp_name = "stack23"
out_dir = "../newdata/Output/"
sub_dir = "../newdata/Output/Subm/"
class Stack1Conf(object):
    name = exp_name
    test = [
        "../newdata/test_basic_30m.csv",
        #sub_dir + 'sub_v20_20180419002702_lgb_lr_0.2_num_leaves_7_max_depth_3.csv',
        #sub_dir + 'sub_wordbatch_fm_ftrl_16_20180417231311.csv',
        #sub_dir + 'sub_wordbatch_fm_ftrl_wordbatch_fm_ftrl_16.csv',
        "sub_pred.v22_20180507200130.csv",
        "sub_pred.v30_20180508032203.csv",
        #sub_dir + 'sub_wordbatch_fm_ftrl_wordbatch_fm_ftrl_16.csv',
        #sub_dir + 'sub_wordbatch_fm_ftrl_wordbatch_fm_ftrl_16.csv',

    ]
    train = [
        "../newdata/train_basic_30m.csv",
        #out_dir + 'cv_v20_20180419002702_lgb_lr_0.2_num_leaves_7_max_depth_3.csv',
        #out_dir + 'cv_wordbatch_fm_ftrl_16_20180417231311.csv',
        #out_dir + 'cv_wordbatch_fm_ftrl_wordbatch_fm_ftrl_16.csv',
        #out_dir + 'cv_pred.Feat@stack11_Learner@reg_fmftrl.csv',
        'val_pred.v22_20180507200130.csv',
        'val_pred.v30_20180508032203.csv',
    ]
    test_predictors = [
        [
            'weight','click_id',
            #'ip','app','device','channel','os','hour',
            'app','device','channel','os','hour',
        ],
        ['is_attributed'],
        ['is_attributed'],
    ]
    train_predictors = [
        [
            'weight','click_id',
            #'ip','app','device','channel','os','hour',
            'app','device','channel','os','hour',
        ],
        ['predicted'],
        ['predicted'],
    ]
    # Number of dimensions to combine
    ndim = 11
    #Target in data
    target_file = "../newdata/train_basic.csv"
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

#main(conf, learner_name, exp_name)
params_dict={
	'learning_rate': 0.052000000000000005, 
	'scale_pos_weight': 100, 
	'boosting_type': 'gbdt', 
	'metric': 'auc', 
	'subsample_for_bin': 200000, 
	'subsample_freq': 1, 
	'min_split_gain': 0, 
	'max_depth': 5, 
	'reg_alpha': 0, 
	'min_child_samples': 100, 
	'nthread': 8, 
	'device': 'gpu', 
	'colsample_bytree': 0.5, 
	'subsample': 0.7, 
	'min_child_weight': 9.340865543712855e-10, 
	'objective': 'binary', 
	'reg_lambda': 0, 
	'num_leaves': 36, 
	'verbose': 1
}
eval(params_dict, conf, learner_name, exp_name)


