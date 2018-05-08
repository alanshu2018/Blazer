#THANK YOU AND ACKNOLEDGEMENTS:
# This kernel develops further the ideas suggested in:
#   *  "lgbm starter - early stopping 0.9539" by Aloisio Dourado, https://www.kaggle.com/aloisiodn/lgbm-starter-early-stopping-0-9539/code
#   * "LightGBM (Fixing unbalanced data)" by Pranav Pandya, https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-auc-0-9787?scriptVersionId=2777211
#   * "LightGBM with count features" by Ravi Teja Gutta, https://www.kaggle.com/rteja1113/lightgbm-with-count-features
# I would like to extend my gratitude to these individuals for sharing their work.

# WHAT IS NEW IN THIS VERSION?
# In addition to some cosmetic changes to the code/LightGBM parameters, I am adding the 'ip' feature to and
# removing the 'day' feature from the training set, and using the last chunk of the training data to build the model.

# What new is NICKS VERSION?
#1 Added Day of Week Time Variable, A IP Count Variable, Feature Importance
#2 Increased validation set to 15%
#3 Imbalanced parameter for lgbm, lower learning rate
#4 new variables- "ip_hour_channel", "ip_hour_os", "ip_hour_app","ip_hour_device"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
from LRTunner import Tunner
from hyperopt import fmin, hp, tpe
import hyperopt
from sklearn.model_selection import *

Ver="v20"
print("Preparing the datasets for training...")

evals_results = {}

print("Training the model...")

#categorical = ['ip','app','device','os','day','hour','hour_4','x1','x2','x3',
#              'x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18']
categorical = [
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

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 255,
    'max_depth': 6,
    'min_child_samples': 100,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'subsample_for_bin': 200000,
    'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'nthread': 8,
    'verbose': 1,
    'is_unbalance': True
    #'scale_pos_weight':99
}

class LGBModel(object):
    def __init__(self,config):
        self.config = config
        self.model = None

    def get_config(self,name,default):
        return self.config.get(name,default)

    def fit_model(self,X_train,y_train,X_test,y_test,W_train=None,W_test=None,max_steps=900,do_predict=True):
        dvalid = lgb.Dataset(X_test, label=y_test,
             feature_name=predictors,
             max_bin =self.get_config("max_bin",150),
             free_raw_data= False if do_predict else True,
             weight=W_test,
             )

        dtrain = lgb.Dataset(X_train, label=y_train,
              feature_name=predictors,
              max_bin =self.get_config("max_bin",150),
              weight=W_train,
              )

        model = lgb.train(self.config,
               dtrain,
               valid_sets=[dvalid,dtrain],
               valid_names=['valid','train'],
               num_boost_round=max_steps,
               early_stopping_rounds=50,
               categorical_feature=categorical,
               verbose_eval=5,
               init_model= None,
               feval=None)
        step = model.best_iteration
        auc = model.best_score['valid']['auc']
        preds = None
        if do_predict:
            preds = model.predict(X_test)

        return model, preds, step, auc

    def fit_cv(self,train_data):
        kf = KFold(n_splits=5)
        X = train_data[predictors].values
        W = train_data['weight'].values
        y = train_data['click_id'].values

        all_preds = np.zeros(len(X))
        aucs = []
        iterations = []
        idx = 0
        for train,test in kf.split(X):
            print("Fit_cv:idx={}".format(idx))
            idx +=1
            X_train, X_test, y_train, y_test = X[train],X[test],y[train],y[test] #train_test_split(X,y,test_size=0.10, random_state=r)
            W_train,W_test = W[train],W[test]

            model, preds, step, auc = self.fit_model(X_train,y_train,X_test,y_test,W_train,W_test,self.config.get("num_boost_round",900))
            all_preds[test] = preds
            aucs.append(auc)
            iterations.append(step)
            gc.collect()

        auc = np.mean(aucs)
        max_step = np.max(iterations)
        print("AUC:aucs={},mean_auc={}".format(aucs,auc))
        print("Step:steps={},max_step={}".format(iterations,max_step))
        return auc, max_step, all_preds

    def fit(self, train_data, dev_data,epochs=None):
        W_train = train_data['weight']
        W_test = dev_data['weight']
        X_train, X_test, y_train, y_test = train_data[predictors].values,dev_data[predictors].values,train_data['click_id'].values,dev_data['click_id'].values
        model, preds, steps, auc = self.fit_model(X_train,y_train,X_test,y_test,W_train,W_test,self.get_config('num_boost_round',900),do_predict=False)
        return auc, steps

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
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

def read_csv(file,nrows=0):
    if nrows <=0:
        train_data = pd.read_csv(file, dtype=dtypes)
    else:
        train_data = pd.read_csv(file, dtype=dtypes,nrows=nrows)
    return train_data

class LGBTunner(Tunner):
    def __init__(self, train_data, dev_data):
        self.train_data = train_data
        self.dev_data = dev_data

    def get_space(self):
        # Searching space
        space = {
            'learning_rate': hp.choice("learning_rate", [1e-2,1e-3,1e-4]), #[0.05,0.01,0.1]),
            #'embed_dim':hp.choice("embed_dim",[20,50]),
            'num_leaves':hp.choice("num_leaves",[32,64,128,256]),
            'max_depth': hp.choice("max_depth",[-1,6]),
        }
        return space

    def build(self, args):
        # args to config
        print("Build with args:{}".format(args))
        config = params
        for k,v in args.items():
            config[k] = v

        if config.get("scale_pos_weight",None) is not None and config.get('is_unbalance',None) is not None:
            del(config["is_unbalance"])
        model = LGBModel(config)
        return model, config

    def train_and_evaluate(self,model,config):
        auc, step = model.fit(self.train_data,self.dev_data)

        auc_error = 1 - auc
        return auc_error, step

    def train(self,model,train_data,dev_data,epochs=None):
        auc, step = model.fit(train_data,dev_data,epochs)
        return model, auc, step

    def train_cv(self,model,train_data):
        auc, step, all_predicts = model.fit_cv(train_data)
        return auc, step, all_predicts

    def train_all_data(self,model,train_data):
        X = train_data[predictors]
        y = train_data['click_id']
        W = train_data['weight'].values
        model, preds, steps, auc = model.fit_model(X,y,X,y,W,W,model.get_config("num_boost_round",900),do_predict=False)
        return model,auc, steps

    def predict_and_save(self,name, model, test_data):
        print("Predicting for name:{}".format(name))
        sub = pd.DataFrame()

        X = test_data[predictors]
        y = test_data['click_id']
        predicted = model.predict(X)
        sub['click_id'] = map(int,y) #.astype('int')
        sub['is_attributed'] = predicted
        print("writing {}......".format(name))
        sub.to_csv(name,index=False)
        del sub
        gc.collect()

import time
import pandas as pd
from DataLoader import DataLoader
import sys

def main(debug=True):
    print("Building model...")
    start = time.time()

    if False:
        # Train on small train, and evaluate on dev1
        train_data = read_csv("newdata/train_{}.csv".format(Ver))
        #test_data = read_csv("newdata/test_{}.csv".format(Ver))
        total_num = len(train_data)
        total_val = int(total_num * 0.1)
        dev_data = train_data[-total_val:]
        train_data = train_data[:-total_val]
        tunner = LGBTunner(train_data, dev_data)
        best_sln = tunner.tune()
        print("best_sln={}".format(best_sln))

    if True:
        args = {
            #'learning_rate': hp.choice("learning_rate", [1e-3,5e-4,1e-4]), #[0.05,0.01,0.1]),
            'learning_rate': 1e-4, #hp.choice("learning_rate", [1e-3,1e-4]), #[0.05,0.01,0.1]),
            #'embed_dim':hp.choice("embed_dim",[20,50]),
            'max_depth':-1, #hp.choice("embed_dim",[20]),
            'num_leaves': 128, #hp.choice("epochs",[10]),
        }
        args = {
            'learning_rate': 0.20,
            #'is_unbalance': 'true', # replaced with scale_pos_weight argument
            'num_leaves': 7,  # 2^max_depth - 1
            'max_depth': 3,  # -1 means no limit
            'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
            'max_bin': 100,  # Number of bucketed bin for feature values
            'subsample': 0.7,  # Subsample ratio of the training instance.
            'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
            'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
            'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'scale_pos_weight':200 # because training data is extremely unbalanced
        }

        arg_str="lgb_lr_{}_num_leaves_{}_max_depth_{}".format(
            args['learning_rate'],
            args['num_leaves'],
            args['max_depth'],
        )

        date_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        data_tag = "%s_%s"%(Ver,date_str)
        print("Build with args:{}".format(args))
        #Use CV to find some parameters and save predicted result
        train_data = read_csv("newdata/train_{}.csv".format(Ver))
        test_data = read_csv("newdata/test_{}.csv".format(Ver))
        tunner = LGBTunner(train_data, test_data)
        model, config = tunner.build(args)

        if True:
            auc, step, all_preds = tunner.train_cv(model,train_data)
            print("***Train First stage Finished:auc={},step={}".format(auc,step))
            print("Save CV predictions")
            df_sub = pd.DataFrame({'predicted': all_preds})
            df_sub.to_csv("cv_lgb_{}.csv".format(data_tag), index=False)
            df_sub.to_csv("sub_" + data_tag + "_" + arg_str+".csv", index=False)
            print("Done!")
            """
            # Nick's Feature Importance Plot
            import matplotlib.pyplot as plt
            lgb_model = model.model
            f, ax = plt.subplots(figsize=[7,10])
            lgb.plot_importance(lgb_model, ax=ax, max_num_features=len(predictors))
            plt.title("Light GBM Feature Importance")
            plt.savefig('feature_import.png')

            # Feature names:
            print('Feature names:', lgb_model.feature_name())
            # Feature importances:
            print('Feature importances:', list(lgb_model.feature_importance()))

            feature_imp = pd.DataFrame(lgb_model.feature_name(),list(lgb_model.feature_importance()))
            """
            #step =309
            args['num_boost_round'] = step
            model, config = tunner.build(args)
            model, auc, step = tunner.train_all_data(model,train_data)
            print("***Train Second stage Finished:auc={},step={}".format(auc,step))

            tunner.predict_and_save("sub_" + data_tag + "_" + arg_str+".csv",model,test_data)

    end = time.time()
    print("Cost {} seconds".format(end - start))


if __name__ == "__main__":
  main()
