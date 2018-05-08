#coding: utf-8

"""
@author: Chenglong Chen <c.chenglong@gmail.com>
@brief: definitions for
        - learner & ensemble learner
        - feature & stacking feature
        - task & stacking task
        - task optimizer
"""

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


logname = "task_%s_%s.log"%("global", time_utils._timestamp())
logger = logging_utils._get_logger(config.LOG_DIR, logname)

class Learner(object):
    def __init__(self, learner_name, param_dict):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = self._get_learner()

    def __str__(self):
        return self.learner_name

    def _get_learner(self):
        # xgboost
        if self.learner_name in ["reg_xgb_linear", "reg_xgb_tree", "reg_xgb_tree_best_single_model"]:
            return LGBMRegressor(**self.param_dict)
        if self.learner_name in ["clf_xgb_linear", "clf_xgb_tree"]:
            return LGBMClassifier(**self.param_dict)
        # sklearn
        if self.learner_name == "reg_skl_lasso":
            return Lasso(**self.param_dict)
        if self.learner_name == "reg_skl_ridge":
            return Ridge(**self.param_dict)
        if self.learner_name == "reg_skl_random_ridge":
            return RandomRidge(**self.param_dict)
        if self.learner_name == "reg_skl_bayesian_ridge":
            return BayesianRidge(**self.param_dict)
        if self.learner_name == "reg_skl_svr":
            return SVR(**self.param_dict)
        if self.learner_name == "reg_skl_lsvr":
            return LinearSVR(**self.param_dict)
        if self.learner_name == "reg_skl_knn":
            return KNNRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_etr":
            return ExtraTreesRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_rf":
            return RandomForestRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_gbm":
            return GradientBoostingRegressor(**self.param_dict)
        if self.learner_name == "reg_skl_adaboost":
            return AdaBoostRegressor(**self.param_dict)
        # keras
        if self.learner_name == "reg_keras_dnn":
            try:
                return KerasDNNRegressor(**self.param_dict)
            except:
                return None
        # rgf
        if self.learner_name == "reg_rgf":
            return RGFRegressor(**self.param_dict)
        # ensemble
        if self.learner_name == "reg_ensemble":
            return EnsembleLearner(**self.param_dict)

        return None

    def fit(self, X, y, feature_names=None):
        if feature_names is not None:
            self.learner.fit(X, y, feature_names)
        else:
            self.learner.fit(X, y)
        return self

    def predict(self, X, feature_names=None):
        if feature_names is not None:
            y_pred = self.learner.predict(X, feature_names)
        else:
            y_pred = self.learner.predict(X)
        # relevance is in [1,3]
        y_pred = np.clip(y_pred, 1., 3.)
        return y_pred

    def plot_importance(self):
        ax = self.learner.plot_importance()
        return ax


class EnsembleLearner(object):
    def __init__(self, learner_dict):
        self.learner_dict = learner_dict

    def __str__(self):
        return "EnsembleLearner"

    def fit(self, X, y):
        for learner_name in self.learner_dict.keys():
            p = self.learner_dict[learner_name]["param"]
            l = Learner(learner_name, p)._get_learner()
            if l is not None:
                self.learner_dict[learner_name]["learner"] = l.fit(X, y)
            else:
                self.learner_dict[learner_name]["learner"] = None
        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0]), dtype=float)
        w_sum = 0.
        for learner_name in self.learner_dict.keys():
            l = self.learner_dict[learner_name]["learner"]
            if l is not None:
                w = self.learner_dict[learner_name]["weight"]
                y_pred += w * l.predict(X)
                w_sum += w
        y_pred /= w_sum
        return y_pred


class Feature(object):

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
        'weight':'float16',
         'ip_tcount':'float16', 'ip_tchan_count':'float16', 'ip_app_count':'float16',
         'ip_app_os_count':'float16', 'ip_app_os_var':'float16',
         'ip_app_channel_var_day':'float16','ip_app_channel_mean_hour':'float16'
    }

    def __init__(self, config, n_fold=5):
        self.config = config
        self.n_fold = n_fold

        self._load_data()

    def __str__(self):
        return self.config.name

    def _load_data(self):
        # Read target
        logger.info("Read target:{}".format(self.config.target_file))
        target_df = pd.read_csv(self.config.target_file,usecols=[self.config.target],dtype=self.dtypes)
        self.y_tr = target_df[self.config.target].values
        del(target_df)
        gc.collect()

        # Read Train data
        train_datas = []
        for file, predictors in zip(self.config.train,self.config.train_predictors):
            logger.info("Read train :{}".format(file))
            df = pd.read_csv(file,usecols=predictors,dtype=self.dtypes)
            d = df[predictors].values
            del(df)
            gc.collect()
            train_datas.append(d)
        self.X_tr = np.concatenate(train_datas,axis=1)

        # Read test data
        test_datas = []
        for file, predictors in zip(self.config.test,self.config.test_predictors):
            logger.info("Read test :{}".format(file))
            df = pd.read_csv(file,usecols=predictors+['click_id'],dtype=self.dtypes)
            d = df[predictors].values
            click_id = df['click_id'].values
            del(df)
            gc.collect()
            test_datas.append(d)
        self.X_te = np.concatenate(test_datas,axis=1)
        self.y_te = click_id

        self.len_train = len(self.X_tr)
        self.len_test = len(self.X_te)
        logger.info("Load data done")

    ## for CV
    def _get_train_valid_data(self):
        kf = KFold(n_splits=self.n_fold)
        for train,test in kf.split(self.X_tr):
            yield (self.X_tr[train],self.y_tr[train],self.X_tr[test],self.y_tr[test],train,test)

    ## for refit
    def _get_train_test_data(self):
        # feature
        return self.X_tr, self.y_tr, self.X_te, self.y_te

    ## for feature importance
    def _get_feature_names(self):
        return self.config.name

class StackingFeature(Feature):
    def __init__(self, feature_name):
        super(StackingFeature,self).__init__(feature_name)

    ## for CV
    def _get_train_valid_data(self):
        return super(StackingFeature,self)._get_train_valid_data()

    ## for refit
    def _get_train_test_data(self, i):
        # feature
        return super(StackingFeature,self)._get_train_test_data()


class Task(object):
    def __init__(self, learner, feature, suffix, logger, verbose=True, plot_importance=False):
        self.learner = learner
        self.feature = feature
        self.suffix = suffix
        self.logger = logger
        self.verbose = verbose
        self.plot_importance = plot_importance
        self.n_iter = 5 #self.feature.n_iter
        self.rmse_cv_mean = 0
        self.rmse_cv_std = 0

    def __str__(self):
        return "[Feat@%s]_[Learner@%s]%s"%(str(self.feature), str(self.learner), str(self.suffix))

    def _print_param_dict(self, d, prefix="      ", incr_prefix="      "):
        for k,v in sorted(d.items()):
            if isinstance(v, dict):
                self.logger.info("%s%s:" % (prefix,k))
                self._print_param_dict(v, prefix+incr_prefix, incr_prefix)
            else:
                self.logger.info("%s%s: %s" % (prefix,k,v))

    def cv(self):
        start = time.time()
        if self.verbose:
            self.logger.info("="*50)
            self.logger.info("Task")
            self.logger.info("      %s" % str(self.__str__()))
            self.logger.info("Param")
            self._print_param_dict(self.learner.param_dict)
            self.logger.info("Result")
            self.logger.info("      Run      RMSE        Shape")

        auc_cv = np.zeros(self.n_iter)
        total_train = self.feature.len_train
        train_pred = np.zeros(total_train)
        i = -1
        for (X_train, y_train, X_valid, y_valid,train_ind,valid_ind) in self.feature._get_train_valid_data():
            i += 1
            print(X_train[:10])
            print(y_train[:10])
            # data
            #X_train, y_train, X_valid, y_valid = self.feature._get_train_valid_data(i)
            # fit
            self.learner.fit(X_train, y_train)
            y_pred = self.learner.predict(X_valid)
            train_pred[valid_ind] = y_pred
            auc_cv[i] = dist_utils._rmse(y_valid, y_pred)
            # log
            self.logger.info("      {:>3}    {:>8}    {} x {}".format(
                i+1, np.round(auc_cv[i],6), X_train.shape[0], X_train.shape[1]))

        # save
        fname = "%s/cv_pred.%s.csv"%(config.OUTPUT_DIR, self.__str__())
        df = pd.DataFrame({"click_id": y_valid, "predicted": y_pred})
        df.to_csv(fname, index=False, columns=["click_id", "predicted"])

        self.rmse_cv_mean = np.mean(auc_cv)
        self.rmse_cv_std = np.std(auc_cv)
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        if self.verbose:
            self.logger.info("AUC")
            self.logger.info("      Mean: %.6f"%self.rmse_cv_mean)
            self.logger.info("      Std: %.6f"%self.rmse_cv_std)
            self.logger.info("Time")
            if _min > 0:
                self.logger.info("      %d mins"%_min)
            else:
                self.logger.info("      %d secs"%_sec)
            self.logger.info("-"*50)
        return self

    def refit(self):
        X_train, y_train, X_test, y_test= self.feature._get_train_test_data()
        if self.plot_importance:
            feature_names = self.feature._get_feature_names()
            self.learner.fit(X_train, y_train, feature_names)
            y_pred = self.learner.predict(X_test, feature_names)
        else:
            self.learner.fit(X_train, y_train)
            y_pred = self.learner.predict(X_test)

        # save
        # submission
        fname = "%s/sub_pred.%s.[Mean%.6f]_[Std%.6f].csv"%(
            config.SUBM_DIR, self.__str__(), self.rmse_cv_mean, self.rmse_cv_std)
        pd.DataFrame({"click_id": y_test, "is_attributed": y_pred}).to_csv(fname, index=False)

        # plot importance
        if self.plot_importance:
            ax = self.learner.plot_importance()
            ax.figure.savefig("%s/%s.pdf"%(config.FIG_DIR, self.__str__()))
        return self

    def go(self):
        self.cv()
        self.refit()
        return self


class StackingTask(Task):
    def __init__(self, learner, feature, suffix, logger, verbose=True, refit_once=False):
        super(StackingTask,self).__init__(learner, feature, suffix, logger, verbose)
        self.refit_once = refit_once

    def refit(self):
        for i in range(self.n_iter):
            if self.refit_once and i >= 1:
                break
            X_train, y_train, X_test,y_test = self.feature._get_train_test_data(i)
            self.learner.fit(X_train, y_train)
            if i == 0:
                y_pred = self.learner.predict(X_test)
            else:
                y_pred += self.learner.predict(X_test)
        if not self.refit_once:
            y_pred /= float(self.n_iter)

        # save
        # submission
        fname = "%s/sub_pred.%s.[Mean%.6f]_[Std%.6f].csv"%(
            config.SUBM_DIR, self.__str__(), self.rmse_cv_mean, self.rmse_cv_std)
        pd.DataFrame({"click_id": y_test, "is_attributed": y_pred}).to_csv(fname, index=False)
        return self


class TaskOptimizer(object):
    def __init__(self, task_mode, learner_name, data_config, logger,
                 max_evals=100, verbose=True, refit_once=False, plot_importance=False):
        self.task_mode = task_mode
        self.learner_name = learner_name
        self.data_config = data_config
        self.feature = self._get_feature()
        self.logger = logger
        self.max_evals = max_evals
        self.verbose = verbose
        self.refit_once = refit_once
        self.plot_importance = plot_importance
        self.trial_counter = 0
        self.model_param_space = ModelParamSpace(self.learner_name)

    def _get_feature(self):
        if self.task_mode == "single":
            feature = Feature(self.data_config)
        elif self.task_mode == "stacking":
            feature = StackingFeature(self.data_config)
        return feature

    def _obj(self, param_dict):
        self.trial_counter += 1
        param_dict = self.model_param_space._convert_int_param(param_dict)
        learner = Learner(self.learner_name, param_dict)
        suffix = "_[Id@%s]"%str(self.trial_counter)
        if self.task_mode == "single":
            self.task = Task(learner, self.feature, suffix, self.logger, self.verbose, self.plot_importance)
        elif self.task_mode == "stacking":
            self.task = StackingTask(learner, self.feature, suffix, self.logger, self.verbose, self.refit_once)
        self.task.go()
        ret = {
            "loss": self.task.rmse_cv_mean,
            "attachments": {
                "std": self.task.rmse_cv_std,
            },
            "status": STATUS_OK,
        }
        return ret

    def run(self):
        start = time.time()
        trials = Trials()
        best = fmin(self._obj, self.model_param_space._build_space(), tpe.suggest, self.max_evals, trials)
        best_params = space_eval(self.model_param_space._build_space(), best)
        best_params = self.model_param_space._convert_int_param(best_params)
        trial_rmses = np.asarray(trials.losses(), dtype=float)
        best_ind = np.argmin(trial_rmses)
        best_rmse_mean = trial_rmses[best_ind]
        best_rmse_std = trials.trial_attachments(trials.trials[best_ind])["std"]
        self.logger.info("-"*50)
        self.logger.info("Best AUC")
        self.logger.info("      Mean: %.6f"%best_rmse_mean)
        self.logger.info("      std: %.6f"%best_rmse_std)
        self.logger.info("Best param")
        self.task._print_param_dict(best_params)
        end = time.time()
        _sec = end - start
        _min = int(_sec/60.)
        self.logger.info("Time")
        if _min > 0:
            self.logger.info("      %d mins"%_min)
        else:
            self.logger.info("      %d secs"%_sec)
        self.logger.info("-"*50)


#------------------------ Main -------------------------
class Stack1Conf(object):
    name = "stack1"
    test = [
        '../sub_v20_20180419002702_lgb_lr_0.2_num_leaves_7_max_depth_3.csv',
        #'../sub_wordbatch_fm_ftrl_16_20180417231311.csv',
        #'../sub_wordbatch_fm_ftrl_wordbatch_fm_ftrl_16.csv',
    ]
    train = [
        '../cv_v20_20180419002702_lgb_lr_0.2_num_leaves_7_max_depth_3.csv',
        #'../cv_wordbatch_fm_ftrl_16_20180417231311.csv',
        #'../cv_wordbatch_fm_ftrl_wordbatch_fm_ftrl_16.csv',
    ]
    test_predictors = [
        ['is_attributed'],
        ['is_attributed'],
        ['is_attributed'],
    ]
    train_predictors = [
        ['predicted'],
        ['predicted'],
        ['predicted'],
    ]
    #Target in data
    target_file = "../newdata/train_v20.csv"
    target = 'click_id'

def main():
    conf = Stack1Conf
    task_mode = "stacking"
    feature_name = conf.name
    learner_name = "clf_xgb_tree"
    max_evals = 10
    refit_once = True
    logname = "[Feat@%s]_[Learner@%s]_hyperopt_%s.log"%(
        feature_name, learner_name, time_utils._timestamp())


    logger = logging_utils._get_logger(config.LOG_DIR, logname)
    optimizer = TaskOptimizer(task_mode, learner_name,
                              conf, logger, max_evals, verbose=True,
                              refit_once=refit_once, plot_importance=False)
    optimizer.run()

"""
def parse_args(parser):
    parser.add_option("-m", "--mode", type="string", dest="task_mode",
                      help="task mode", default="single")
    parser.add_option("-f", "--feat", type="string", dest="feature_name",
                      help="feature name", default="basic")
    parser.add_option("-l", "--learner", type="string", dest="learner_name",
                      help="learner name", default="reg_skl_ridge")
    parser.add_option("-e", "--eval", type="int", dest="max_evals",
                      help="maximun number of evals for hyperopt", default=100)
    parser.add_option("-o", default=False, action="store_true", dest="refit_once",
                      help="stacking refit_once")
    parser.add_option("-p", default=False, action="store_true", dest="plot_importance",
                      help="plot feautre importance (currently only for xgboost)")

    (options, args) = parser.parse_args()
    return options, args
"""

if __name__ == "__main__":
    #parser = OptionParser()
    #options, args = parse_args(parser)
    #main(options)
    main()