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
from model_param_space import *

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

int_params = [
    "num_round", "n_estimators", "min_samples_split", "min_samples_leaf",
    "n_neighbors", "leaf_size", "seed", "random_state", "max_depth", "degree",
    "hidden_units", "hidden_layers", "batch_size", "nb_epoch", "dim", "iter",
    "factor", "iteration", "n_jobs", "max_leaf_forest", "num_iteration_opt",
    "num_tree_search", "min_pop", "opt_interval",'num_leaves'
]
int_params = set(int_params)

## xgboost
xgb_random_seed = config.RANDOM_SEED
xgb_nthread = config.NUM_CORES
xgb_n_estimators_min = 100
xgb_n_estimators_max = 1000
xgb_n_estimators_step = 10

## sklearn
skl_random_seed = config.RANDOM_SEED
skl_n_jobs = config.NUM_CORES
skl_n_estimators_min = 50
skl_n_estimators_max = 300
skl_n_estimators_step = 20

class BaseLearner(object):
    name = "base"
    param_space = {}
    def __init__(self):
        #self.param_space = self._convert_int_param(self.param_space)
        pass

    def __str__(self):
        return "BasicLearner"

    def _convert_int_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k,v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_int_param(v[i])
                elif isinstance(v, dict):
                    self._convert_int_param(v)
        return param_dict

    def create_model(self, params):
        return None

    def fit(self, model, X, y, weight=None):
        try:
            model.fit(X, y, weight=weight)
        except:
            model.fit(X,y)
        return model

    def predict(self, model, X, weight=None):
        try:
            return model.predict(X, weight=weight)
        except:
            return model.predict(X)


class LGBMClassifierLearner(BaseLearner):
    name = "clf_lgb"

    param_space = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': hp.qloguniform("learning_rate",np.log(0.002),np.log(0.2),0.002),
        'num_leaves': hp.quniform("num_leaves",0,255,16),
        'max_depth': hp.choice("max_depth",[-1,3,5,7,10]),
        'min_child_samples': 100,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': hp.quniform("colsample_bytree",0.3,1,0.1),
        'min_child_weight': hp.loguniform("min_child_weight",np.log(1e-10),np.log(1e2)),
        'subsample_for_bin': 200000,
        'min_split_gain': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'nthread': 8,
        'verbose': 5,
        'scale_pos_weight':hp.choice("scale_pos_weight",[99.,99.7,200.]),
        #'is_unbalance': True
    }

    def create_model(self,params):
        self.params = params
        return LGBMClassifier(**params)

# -------------------------------------- Sklearn ---------------------------------------------
## lasso
class LassoRegressorLearner(BaseLearner):
    name = "reg_lasso"
    param_space = {
        "alpha": hp.loguniform("alpha", np.log(0.00001), np.log(0.1)),
        "normalize": hp.choice("normalize", [True, False]),
        "random_state": skl_random_seed
    }

    def create_model(self,params):
        self.params = params
        return Lasso(**params)

## ridge regression
class RidgeRegressorLearner(BaseLearner):
    name = "reg_ridge"
    param_space = {
        "alpha": hp.loguniform("alpha", np.log(0.01), np.log(20)),
        "normalize": hp.choice("normalize", [True, False]),
        "random_state": skl_random_seed
    }

    def create_model(self,params):
        self.params = params
        return Ridge(**params)

## Bayesian Ridge Regression
class BayesianRidgeRegressorLearner(BaseLearner):
    name = "reg_bayes_ridge"
    param_space = {
        "alpha_1": hp.loguniform("alpha_1", np.log(1e-10), np.log(1e2)),
        "alpha_2": hp.loguniform("alpha_2", np.log(1e-10), np.log(1e2)),
        "lambda_1": hp.loguniform("lambda_1", np.log(1e-10), np.log(1e2)),
        "lambda_2": hp.loguniform("lambda_2", np.log(1e-10), np.log(1e2)),
        "normalize": hp.choice("normalize", [True, False])
    }

    def create_model(self,params):
        self.params = params
        return BayesianRidge(**params)

## random ridge regression
class RandomRidgeRegressorLearner(BaseLearner):
    name = "reg_random_ridge"
    param_space = {
        "alpha": hp.loguniform("alpha", np.log(0.01), np.log(20)),
        "normalize": hp.choice("normalize", [True, False]),
        "poly": hp.choice("poly", [False]),
        "n_estimators": hp.quniform("n_estimators", 2, 50, 2),
        "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
        "bootstrap": hp.choice("bootstrap", [True, False]),
        "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
        "random_state": skl_random_seed
    }

    def create_model(self,params):
        self.params = params
        return RandomRidge(**params)

## linear support vector regression
class  LinearSVRRegressorLearner(BaseLearner):
    name = "reg_linear_svr"
    param_space = {
        "normalize": hp.choice("normalize", [True, False]),
        "C": hp.loguniform("C", np.log(1), np.log(100)),
        "epsilon": hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),
        "loss": hp.choice("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]),
        "random_state": skl_random_seed,
    }

    def create_model(self,params):
        self.params = params
        return LinearSVR(**params)

## support vector regression
class  SVRRegressorLearner(BaseLearner):
    name = "reg_svr"
    param_space = {
        "normalize": hp.choice("normalize", [True]),
        "C": hp.loguniform("C", np.log(1), np.log(1)),
        "gamma": hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
        "degree": hp.quniform("degree", 1, 3, 1),
        "epsilon": hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),
        "kernel": hp.choice("kernel", ["rbf", "poly"])
    }
    def create_model(self,params):
        self.params = params
        return SVR(**params)


## K Nearest Neighbors Regression
class KNNRegressorLearner(BaseLearner):
    name = "reg_knn"
    param_space = {
        "normalize": hp.choice("normalize", [True, False]),
        "n_neighbors": hp.quniform("n_neighbors", 1, 20, 1),
        "weights": hp.choice("weights", ["uniform", "distance"]),
        "leaf_size": hp.quniform("leaf_size", 10, 100, 10),
        "metric": hp.choice("metric", ["cosine", "minkowski"][1:]),
    }
    def create_model(self,params):
        self.params = params
        return KNNRegressor(**params)

## extra trees regressor
class ExtraTreeRegressorLearner(BaseLearner):
    name = "reg_etr"
    param_space = {
        "n_estimators": hp.quniform("skl_etr__n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
        "max_features": hp.quniform("skl_etr__max_features", 0.1, 1, 0.05),
        "min_samples_split": hp.quniform("skl_etr__min_samples_split", 1, 15, 1),
        "min_samples_leaf": hp.quniform("skl_etr__min_samples_leaf", 1, 15, 1),
        "max_depth": hp.quniform("skl_etr__max_depth", 1, 10, 1),
        "random_state": skl_random_seed,
        "n_jobs": skl_n_jobs,
        "verbose": 0,
    }

    def create_model(self,params):
        self.params = params
        return ExtraTreesRegressor(**params)

## random forest regressor
class RandomForestRegressorLearner(BaseLearner):
    name = "reg_rf"
    param_space = {
        "n_estimators": hp.quniform("skl_rf__n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
        "max_features": hp.quniform("skl_rf__max_features", 0.1, 1, 0.05),
        "min_samples_split": hp.quniform("skl_rf__min_samples_split", 1, 15, 1),
        "min_samples_leaf": hp.quniform("skl_rf__min_samples_leaf", 1, 15, 1),
        "max_depth": hp.quniform("skl_rf__max_depth", 1, 10, 1),
        "random_state": skl_random_seed,
        "n_jobs": skl_n_jobs,
        "verbose": 0,
    }

    def create_model(self,params):
        self.params = params
        return RandomForestRegressor(**params)

## gradient boosting regressor
class GBMRegressorLearner(BaseLearner):
    name = "reg_gbm"
    param_space = {
        "n_estimators": hp.quniform("skl_gbm__n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
        "learning_rate" : hp.qloguniform("skl__gbm_learning_rate", np.log(0.002), np.log(0.1), 0.002),
        "max_features": hp.quniform("skl_gbm__max_features", 0.1, 1, 0.05),
        "max_depth": hp.quniform("skl_gbm__max_depth", 1, 10, 1),
        "min_samples_leaf": hp.quniform("skl_gbm__min_samples_leaf", 1, 15, 1),
        "random_state": skl_random_seed,
        "verbose": 0,
    }
    def create_model(self,params):
        self.params = params
        return GradientBoostingRegressor(**params)


## adaboost regressor
class AdaBoostRegressorLearner(BaseLearner):
    name = "reg_adaboost"
    param_space = {
        "base_estimator": hp.choice("base_estimator", ["dtr", "etr"]),
        "n_estimators": hp.quniform("n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
        "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
        "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
        "max_depth": hp.quniform("max_depth", 1, 10, 1),
        "loss": hp.choice("loss", ["linear", "square", "exponential"]),
        "random_state": skl_random_seed,
    }

    def create_model(self,params):
        self.params = params
        return AdaBoostRegressor(**params)

# -------------------------------------- Keras ---------------------------------------------
## regression with Keras' deep neural network
class KerasDNNRegressorLearner(BaseLearner):
    name = "reg_keras_dnn"
    param_space = {
        "input_dropout": hp.quniform("input_dropout", 0, 0.2, 0.05),
        "hidden_layers": hp.quniform("hidden_layers", 1, 3, 1),
        "hidden_units": hp.quniform("hidden_units", 32, 128, 32),
        "hidden_activation": hp.choice("hidden_activation", ["prelu", "relu", "elu"]),
        "hidden_dropout": hp.quniform("hidden_dropout", 0, 0.5, 0.05),
        "batch_norm": hp.choice("batch_norm", ["before_act", "after_act", "no"]),
        "optimizer": hp.choice("optimizer", ["adam", "adadelta", "rmsprop"]),
        "batch_size": hp.quniform("batch_size", 16, 128, 16),
        "nb_epoch": hp.quniform("nb_epoch", 1, 20, 1),
    }

    def create_model(self,params):
        self.params = params
        return KerasDNNRegressor(**params)

# -------------------------------------- RGF ---------------------------------------------
class RGFRegressorLearner(BaseLearner):
    name = "reg_rgf"
    param_space = {
        "reg_L2": hp.loguniform("reg_L2", np.log(0.1), np.log(10)),
        "reg_sL2": hp.loguniform("reg_sL2", np.log(0.00001), np.log(0.1)),
        "max_leaf_forest": hp.quniform("max_leaf_forest", 10, 1000, 10),
        "num_iteration_opt": hp.quniform("num_iteration_opt", 5, 20, 1),
        "num_tree_search": hp.quniform("num_tree_search", 1, 10, 1),
        "min_pop": hp.quniform("min_pop", 1, 20, 1),
        "opt_interval": hp.quniform("opt_interval", 10, 200, 10),
        "opt_stepsize": hp.quniform("opt_stepsize", 0.1, 1.0, 0.1)
    }

    def create_model(self,params):
        self.params = params
        return RGFRegressor(**params)

