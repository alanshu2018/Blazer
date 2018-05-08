"""
I thank Pranav Pandya for the script - https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-auc-0-9787

The training data for this kernel was prepared by selecting only the records from 9th of November.

The test dataset contains significant data for only the following hours of 10 Nov - 4,5,9,10,13,14. Check my EDA for the same - https://www.kaggle.com/gopisaran/indepth-eda-entire-talkingdata-dataset
So I further filtered the Nov 9th data using those hours.

Used the following SQL to prepare training sample - fraud_sample.csv which consists of only 18 million records.

select ip,app,device,os,channel,click_time,attributed_time,is_attributed from fraud where extract(day from click_time)=9 and
extract(hour from click_time) in (4,5,9,10,13,14)
LB: 0.9680

"""

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc

def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                      feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgtrain, xgvalid],
                     valid_names=['train','valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1

path = 'data/'


print('loading train data...')
#train_df = pd.read_hdf(path+"train.hdf","data",start= 131886954)
train_df = pd.read_hdf(path+"train.hdf","data")
#train_df = pd.read_csv(path+"fraudsampleprepared/fraud_sample.csv")
#train_df.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']

print('loading test data...')
#print(train_df.head())
#test_df = pd.read_csv(path+"talkingdata-adtracking-fraud-detection/test.csv")
test_df = pd.read_hdf(path+"test.hdf","data")
#print(test_df.head())
#train_df=train_df.append(test_df)

#del test_df
#gc.collect()

VALID_MODE=True
GEN_ROUND=240
gen_name="noip_gen_new_valid"


max_num_rounds = 350
if len(gen_name) == 0:
    gen_name = "1"

all_train_df = train_df[
    (train_df.hour==4)|(train_df.hour==5)| \
    (train_df.hour==9)|(train_df.hour==10) \
    |(train_df.hour==13)|(train_df.hour==14)
]
len_train = len(all_train_df)
val_df = all_train_df[(len_train-3000000):len_train]
train_df = all_train_df[:(len_train-3000000)]

if VALID_MODE == False:
    max_num_rounds = GEN_ROUND
    train_df = all_train_df

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'day', 'wday',
              #'qty_hour', 'ip_app_count', 'ip_app_os_count',
              "ip_os_dev_count","ip_app_dev_count","ip_app_count","ip_app_wday_hour_count", #4
              "ip_app_day_hour_count","ip_app_hour_count","ip_dev_count", #3
              "qty_wday_hour","qty_day_hour","qty_hour","app_os_count", #4
              "app_count","app_os_dev_count","app_dev_count","ip_app_os_count" #4
              ]
categorical = ['app', 'device', 'os', 'channel', 'hour', 'day', 'wday',
               "ip_os_dev_count","ip_app_dev_count","ip_app_count","ip_app_wday_hour_count", #4
               "ip_app_day_hour_count","ip_app_hour_count","ip_dev_count", #3
               "qty_wday_hour","qty_day_hour","qty_hour","app_os_count", #4
               "app_count","app_os_dev_count","app_dev_count","ip_app_os_count" #4
               ]

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

gc.collect()

print("Training...")
start_time = time.time()


params = {
    'learning_rate': 0.1,#0.15
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 15,  # 2^max_depth - 1
    'max_depth': 4,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': .7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99, # because training data is extremely unbalanced
    #'reg_lambda': 10,  # L1 regularization term on weights
}
bst = lgb_modelfit_nocv(params,
                        train_df,
                        val_df,
                        predictors,
                        target,
                        objective='binary',
                        metrics='auc',
                        early_stopping_rounds=50,
                        verbose_eval=True,
                        num_boost_round=max_num_rounds,
                        categorical_features=categorical)

print('[{}]: model training time'.format(time.time() - start_time))
del train_df
del val_df
gc.collect()

if VALID_MODE == False:
    print("Predicting...")
    sub['is_attributed'] = bst.predict(test_df[predictors])
    print("writing...")
    sub.to_csv('sub_lgb_balanced99_{}.csv'.format(gen_name),index=False)
    print("done...")