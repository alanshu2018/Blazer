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


click_means=np.array([0]*8+[1.866671e+07,5.083459e+04,2.351688e+07,1.809987e+08,4.543603e+06,
             1.215442e+05,2.213557e+06,9.366850e+05,1.703376e+07,1.928687e+05,
             3.039419e+06,3.092410e+02,5.026361e+03,1.170573e+06,2.202677e+07,
             ]+[0.0] *30)
click_std=np.array([0]*8 +[ 1.147628e+07, 1.801862e+05, 2.076854e+07, 4.308380e+07, 3.937021e+06,
            1.535289e+05, 2.650926e+06, 6.892881e+05, 1.136252e+07, 2.888697e+05,
            3.759831e+06, 1.343773e+03, 2.044271e+04, 1.112249e+06, 2.028386e+07,
            ]+[0.0]*30)
ver="v19"
print("Preparing the datasets for training...")

evals_results = {}

print("Training the model...")

#categorical = ['ip','app','device','os','day','hour','hour_4','x1','x2','x3',
#              'x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18']
categorical = [
    'ip','app','device','channel','os','day','hour','hour_4'
    ]
predictors = [
                'ip','app','device','channel','os','day','hour','hour_4',
                'c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15',
                'r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15',
                'h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15',
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

class LGBModel(object):
    def __init__(self,config):
        self.config = config
        self.model = None

    def fit1(self, train_data, dev_data,epochs=None):
        val_x = None
        val_y = None
        for (dev_data_x,dev_data_y) in dev_data.get_minibatch(batch_size=80000000):
            val_x = dev_data_x
            val_y = dev_data_y
            break

        val_x = (val_x - click_means)/(click_std + 1e-8)
        dvalid = lgb.Dataset(val_x, label=val_y,
                     feature_name=predictors,
                     max_bin =150,
                     #free_raw_data= False,
                     )

        gc.collect()

        X = []
        Y = []
        for (x,y) in train_data.get_minibatch(batch_size=30000000):
            X.append(x)
            Y.append(y)

        X_train = np.concatenate(X, axis=0)
        Y_train = np.concatenate(Y,axis=0)

        X_train = (X_train - click_means)/(click_std + 1e-8)
        dtrain = lgb.Dataset(X_train, label=Y_train,
                         feature_name=predictors,
                         max_bin =150,
                         )
        del X
        del Y
        del X_train
        del Y_train
        del val_x
        del val_y
        gc.collect()

        self.model = lgb.train(self.config,
                          dtrain,
                          valid_sets=[dvalid],
                          valid_names=['valid'],
                          num_boost_round=900,
                          early_stopping_rounds=30,
                          categorical_feature=categorical,
                          verbose_eval=5,
                          init_model= self.model,
                          feval=None)

        step = self.model.best_iteration
        auc = self.model.best_score['valid']['auc']
        return auc, step

    def fit(self, train_data, dev_data,epochs=None):
        val_x = None
        val_y = None
        for (dev_data_x,dev_data_y) in dev_data.get_minibatch(batch_size=80000000):
            val_x = dev_data_x
            val_y = dev_data_y
            break

        #val_x = (val_x - click_means)/(click_std + 1e-8)
        dvalid = lgb.Dataset(val_x, label=val_y,
                     feature_name=predictors,
                     max_bin =150,
                     free_raw_data= False,
                     )

        gc.collect()

        for (x,y) in train_data.get_minibatch(batch_size=30000000):
            dtrain = lgb.Dataset(x, label=y,
                         feature_name=predictors,
                         max_bin =150,
                         )


            self.model = lgb.train(self.config,
                           dtrain,
                           valid_sets=[dvalid,dtrain],
                           valid_names=['valid','train'],
                           num_boost_round=900,
                           early_stopping_rounds=50,
                           categorical_feature=categorical,
                           verbose_eval=5,
                           init_model= self.model,
                           feval=None)
            del x
            del y
            gc.collect()

        step = self.model.best_iteration
        auc = self.model.best_score['valid']['auc']
        return auc, step

    def fit_split(self, train_data):
        X = []
        y = []
        for (xx, yy) in train_data.get_minibatch(batch_size=10000000,epochs=1):
            X.append(xx)
            y.append(yy)

        X = np.concatenate(X,axis=0)
        y = np.concatenate(y,axis=0)

        train_x,val_x, train_y, val_y = train_test_split(X,y, test_size=0.1)

        dvalid = lgb.Dataset(val_x, label=val_y,
                     feature_name=predictors,
                     max_bin =150,
                     free_raw_data= False,
                     )

        dtrain = lgb.Dataset(train_x, label=train_y,
                     feature_name=predictors,
                     max_bin =150,
                     )
        self.model = lgb.train(self.config,
                      dtrain,
                      valid_sets=[dvalid,dtrain],
                      valid_names=['valid','train'],
                      num_boost_round=900,
                      early_stopping_rounds=80,
                      categorical_feature=categorical,
                      verbose_eval=5,
                      init_model= self.model,
                      feval=None)

        step = self.model.best_iteration
        auc = self.model.best_score['valid']['auc']
        return auc, step

    def predict(self, test_data):
        for (x,y) in test_data.get_minibatch(batch_size=80000000):
            return self.model.predict(x, num_iteration=self.model.best_iteration),y

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

        model = LGBModel(config)
        return model, config

    def train_and_evaluate(self,model,config):
        auc, step = model.fit(self.train_data,self.dev_data)

        auc_error = 1 - auc
        return auc_error, step

    def train(self,model,train_data,dev_data,epochs=None):
        auc, step = model.fit(train_data,dev_data,epochs)
        return model, auc, step

    def train_split(self,model,train_data):
        # Split train data and do the train
        auc, step = model.fit_split(train_data)
        return model, auc, step

    def predict_and_save(self,name, model, test_data):
        print("Predicting for name:{}".format(name))
        sub = pd.DataFrame()
        predicted, labels = model.predict(test_data)
        sub['click_id'] = map(int,labels) #.astype('int')
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

    train_data_loader = DataLoader(files=[
        #("data/dev1_feature_norm.simple.{}.csv".format(ver),2000868),
        #("data/dev2_feature_norm.simple.{}.csv".format(ver),1999132),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv".format(ver),180903890),
        ("newdata/train_small_{}.csv".format(ver),50000000),
    ])
    train_data_loader1 = DataLoader(files=[
        #("data/dev2_feature_norm.simple.{}.csv".format(ver),1999132),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv".format(ver),180903890),
        ("newdata/train_small_{}.csv".format(ver),50000000),
        ("newdata/dev1_{}.csv".format(ver),2000000),
        #("data/train_feature_norm.simple.small.{}.csv".format(ver),30000000),
    ])
    train_data_loader2 = DataLoader(files=[
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.small.{}.csv".format(ver),30000000),
        ("newdata/train_small_{}.csv".format(ver),50000000),
        ("newdata/dev1_{}.csv".format(ver),2000000),
        ("newdata/dev2_{}.csv".format(ver),2000000),
    ])

    dev1_data_loader = DataLoader(files=[
        #("data/dev1_feature_norm.simple.{}.csv".format(ver),2000868),
        ("newdata/dev1_{}.csv".format(ver),2000000),
        #("data/dev2_feature_norm.simple.{}.csv".format(ver),1999132),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv".format(ver),180903890),
    ])

    dev2_data_loader = DataLoader(files=[
        #("data/dev1_feature_norm.simple.{}.csv".format(ver),2000868),
        ("newdata/dev2_{}.csv".format(ver),2000000),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv".format(ver),180903890),
    ])

    dev3_data_loader = DataLoader(files=[
        ("newdata/dev1_{}.csv".format(ver),2000000),
        ("newdata/dev2_{}.csv".format(ver),2000000),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv".format(ver),180903890),
    ])

    test_data_loader = DataLoader(files=[
        #("data/dev1_feature_norm.simple.{}.csv".format(ver),2000868),
        #("data/dev2_feature_norm.simple.{}.csv".format(ver),1999132),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        ("newdata/test_{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv",180903890),
    ])

    if True:
        # Train on small train, and evaluate on dev1
        tunner = LGBTunner(train_data_loader, dev1_data_loader)
        best_sln = tunner.tune()
        print("best_sln={}".format(best_sln))

    if False:
        args = {
            #'learning_rate': hp.choice("learning_rate", [1e-3,5e-4,1e-4]), #[0.05,0.01,0.1]),
            'learning_rate': 1e-4, #hp.choice("learning_rate", [1e-3,1e-4]), #[0.05,0.01,0.1]),
            #'embed_dim':hp.choice("embed_dim",[20,50]),
            'max_depth':-1, #hp.choice("embed_dim",[20]),
            'num_leaves': 128, #hp.choice("epochs",[10]),
        }
        arg_str="lgb_lr_{}_num_leaves_{}_max_depth_{}".format(
            args['learning_rate'],
            args['num_leaves'],
            args['max_depth'],
        )
        date_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        data_tag = "%s_%s"%(ver,date_str)
        print("Build with args:{}".format(args))
        tunner = LGBTunner(train_data_loader, dev1_data_loader)
        model, config = tunner.build(args)

        if True:

            # Train on train + dev1, predict on dev2
            model, auc, step = tunner.train(model,train_data_loader1,dev2_data_loader)
            print("***Train First stage Finished:auc={},step={}".format(auc,step))

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


            tunner.predict_and_save("dev_" +data_tag + "_" + arg_str,model,dev2_data_loader)

            # Train on all the dev data again, and generate the submission
            #model,auc, step = tunner.train_split(model,dev3_data_loader)
            #print("***Train Second stage Finished:auc={},step={}".format(auc,step))
            tunner.predict_and_save("sub_" + data_tag + "_" + arg_str,model,test_data_loader)

    end = time.time()
    print("Cost {} seconds".format(end - start))


if __name__ == "__main__":
  main()
