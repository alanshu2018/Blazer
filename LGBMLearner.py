#coding:utf-8

import numpy as np
from Learner import BaseLearner
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

categoricals = [
    'ip','app','device','channel','os','day','hour'
    ]
predictors = [
        'ip','app','device','os', 'channel', 'hour', 'day',
        'next_click','next_click_shift',
        'X0','X1','X2','X3','X4','X5','X6','X7','X8',
        'ip_tcount', 'ip_tchan_count', 'ip_app_count',
        'ip_app_os_count', 'ip_app_os_var',
        'ip_app_channel_var_day','ip_app_channel_mean_hour'
]

class LightGBMRegressor(object):
    def __init__(self,config, predictors="auto", categoricals="auto"):
        self.config = config
        self.predictors = predictors
        self.categoricals = categoricals
        self._build()

    def _build(self):
        self.model = None

    def get_config(self,name,default):
        return self.config.get(name,default)

    def fit_model(self,X_train,y_train,X_test,y_test,do_validate=True,W_train=None,W_test=None,max_steps=3000,do_predict=True):

        dtrain = lgb.Dataset(X_train, label=y_train,
              feature_name=self.predictors,
              #max_bin =self.get_config("max_bin",150),
              weight=W_train,
              )

        if do_validate:
            dvalid = lgb.Dataset(X_test, label=y_test,
                             feature_name=self.predictors,
                             #max_bin =self.get_config("max_bin",150),
                             free_raw_data= False if do_predict else True,
                             weight=W_test,
                             )
            valid_sets = [dvalid,dtrain]
            valid_names = ['valid','train']
        else:
            dvalid = dtrain
            valid_sets = [dtrain]
            valid_names = ['valid']

        model = lgb.train(self.config,
               dtrain,
               valid_sets=valid_sets, #[dvalid,dtrain],
               valid_names=valid_names,#['valid','train'],
               num_boost_round=max_steps,
               early_stopping_rounds=50,
               categorical_feature=self.categoricals,
               verbose_eval=5,
               init_model= None,
               feval=None)
        step = model.best_iteration
        auc = model.best_score['valid']['auc']
        preds = None
        if do_predict:
            preds = model.predict(X_test)

        return model, preds, step, auc

    def fit(self,X, y, validate=True):
        W_train = None
        W_test = None
        max_steps = self.get_config("num_boost_round",900)
        if validate:
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10, random_state=1024)
            model, preds, step, auc = self.fit_model(X_train,y_train,X_test,y_test,validate,W_train,W_test, max_steps=max_steps,do_predict=False)
        else:
            model, preds, step, auc = self.fit_model(X,y,None,None,validate,W_train,W_test, max_steps=max_steps,do_predict=False)

        print("step:{},auc:{}".format(step,auc))
        self.model = model
        self.best_step = step

    def predict(self,X):
        return self.model.predict(X)

class LightGBMLearner(BaseLearner):
    name = "reg_lgb_native"

    param_space = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': hp.qloguniform("learning_rate",np.log(0.005),np.log(0.2),0.002),
        'num_leaves': hp.quniform("num_leaves",4,255,2),
        'max_depth': hp.choice("max_depth",[-1,3,5,7,10]),
        'min_child_samples': 100,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': hp.quniform("colsample_bytree",0.3,1,0.1),
        'min_child_weight': hp.loguniform("min_child_weight",np.log(1e-10),np.log(1e2)),
        'subsample_for_bin': 200000,
        'min_split_gain': 0,
        'reg_alpha': 0,
        'device': 'gpu',
        'reg_lambda': 0,
        'nthread': 8,
        'verbose': 1,
        #'is_unbalance': True
        'scale_pos_weight':hp.choice("scale_pos_weight",[99,99.7,100,200,300,400])
    }

    def create_model(self,params):
        self.params = params

        return LightGBMRegressor(params)

