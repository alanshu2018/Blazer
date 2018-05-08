"""
If you find this kernel helpful please upvote. Also any suggestion for improvement will be warmly welcomed.
I made cosmetic changes in the [code](https://www.kaggle.com/aharless/kaggle-runnable-version-of-baris-kanber-s-lightgbm/code).
Added some new features. Ran for 25mil chunk rows.
Also taken ideas from various public kernels.
"""
import gc
import time

train_file="../newdata/train_v30.csv"
valid_file="../newdata/valid_v30.csv"
test_file ="../newdata/test_v30.csv"


param_dict={
    'boosting_type': 'gbdt',
    'colsample_bytree': 0.5,
    'learning_rate': 0.012,
    'max_depth': 7,
    'metric': 'auc',
    'min_child_samples': 100,
    'min_child_weight': 0.04214502460126984,
    'min_split_gain': 0,
    'nthread': 8,
    'num_leaves': 78,
    'objective': 'binary',
    'reg_alpha': 0,
    'reg_lambda': 0,
    'scale_pos_weight': 200,
    'subsample': 0.7,
    'subsample_for_bin': 200000,
    'subsample_freq': 1,
    'verbose': 1
}

from LGBMLearner import *

import pandas as pd
import numpy as np
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
        'XX0':'uint16',
        'XX1':'uint16',
        'XX2':'uint16',
        'XX3':'uint16',
        'XX4':'uint16',
        'XX5':'uint16',
        'XX6':'uint16',
        'XX7':'uint16',
        'XX8':'uint16',
        'ip_tchan_var':'float16',
        'ip_app_os_var':'float16',
        'ip_app_channel_var_day':'float16',
        'ip_app_channel_mean_hour':'float16',
        'weight':'float16',
        'ip_tcount':'float16',
        'ip_tchan_count':'float16',
        'ip_app_count':'float16',
        'ip_app_os_count':'float16',
        'ip_app_os_var':'float16',
        'ip_app_channel_var_day':'float16',
        'ip_app_channel_mean_hour':'float16',
        'ip_d_os_c_app':'uint32',
        'ip_c_os':'uint32',
        'ip_cu_c':'uint16',
        'ip_da_cu_h':'uint16',
        'ip_cu_app':'uint16',
        'ip_app_cu_os':'uint16',
        'ip_cu_d':'uint16',
        'app_cu_chl':'uint16',
        'ip_d_os_cu_app':'uint16',
        'ip_da_co':'uint16',
        'ip_app_co':'uint16',
        'ip_app_os_co':'uint16',
        'ip_d_co':'uint16',
        'app_chl_co':'uint16',
        'ip_ch_co':'uint16',
        'ip_app_chl_co':'uint16',
        'app_d_co':'uint16',
        'app_os_co':'uint16',
        'ip_os_co':'uint16',
        'ip_d_os_co':'uint16',
        'ip_app_h_co':'uint16',
        'ip_app_os_h_co':'uint16',
        'ip_d_h_co':'uint16',
        'app_chl_h_co':'uint16',
        'ip_ch_h_co':'uint16',
        'ip_app_chl_h_co':'uint16',
        'app_d_h_co':'uint16',
        'app_os_h_co':'uint16',
        'ip_os_h_co':'uint16',
        'ip_d_os_h_co':'uint16',
        'ip_da_chl_var_h':'float16',
        'ip_chl_var_h':'float16',
        'ip_app_os_var_h':'float16',
        'ip_app_chl_var_da':'float16',
        'ip_app_chl_var_h':'float16',
        'app_os_var_da':'float16',
        'app_d_var_h':'float16',
        'app_chl_var_h':'float16',
        'ip_app_chl_mean_h':'float16',
        'ip_chl_mean_h':'float16',
        'ip_app_os_mean_h':'float16',
        'ip_app_mean_h':'float16',
        'app_os_mean_h':'float16',
        'app_mean_var_h':'float16',
        'app_chl_mean_h':'float16',
        'ip_channel_prevClick':'float16',
        'ip_os_prevClick':'float16',
        'ip_app_device_os_channel_nextClick':'float16',
        'ip_os_device_nextClick':'float16',
        'ip_os_device_app_nextClick':'float16',
    }

def lgb_modelfit_nocv(bst, params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                      feval=None, early_stopping_rounds=50, num_boost_round=900, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.05,
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
    del dtrain
    del dvalid
    gc.collect()

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[ xgvalid],
                     valid_names=['valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     init_model= bst,
                     feval=feval)

    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    #print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return (bst1,bst1.best_iteration)

chunk_size=30000000
def get_chunk_data(file,chunk_size=chunk_size,nrows=-1):
    rcount = 0

    for df_c in pd.read_csv(file, engine='c',
            chunksize=chunk_size,
            usecols=predictors+['click_id'],
            dtype=dtypes): #usecols=usecols):

        idx = rcount
        rcount += len(df_c)
        # Return idx, fold_num and data, labels, weight
        yield df_c #, df_c[self.predictors],df_c[self.target_name], df_c[self.weight_name])

        if nrows >0 and rcount > nrows:
            break

target = 'click_id'
categorical = ['ip','app', 'device', 'os', 'channel', 'hour']

predictors=[
    'ip','app','device','channel','os','hour',
    'ip_d_os_c_app','ip_cu_c','ip_da_cu_h','ip_cu_app',
    'ip_app_cu_os','ip_cu_d','ip_d_os_cu_app','ip_da_co','ip_app_co','ip_d_co','app_chl_co','ip_ch_co',
    'app_d_co','ip_os_co','ip_d_os_co','ip_app_h_co','ip_app_os_h_co','ip_d_h_co','app_chl_h_co','ip_ch_h_co',
    'app_d_h_co','app_os_h_co','ip_os_h_co','ip_d_os_h_co','ip_da_chl_var_h','ip_chl_var_h','ip_app_os_var_h',
    'ip_app_chl_var_da','ip_app_chl_var_h','app_chl_var_h','ip_app_os_mean_h','app_os_mean_h',
    'ip_channel_prevClick','ip_app_device_os_channel_prevClick','ip_os_device_app_prevClick',
    'ip_app_device_os_channel_nextClick','ip_os_device_nextClick',
    'ip_os_device_app_nextClick',
    ]

print("Read valid data")
val_df = pd.read_csv(valid_file,engine='c',usecols=predictors+['click_id'],dtype=dtypes)
print("valid.len={}".format(len(val_df)))

print("************* Train with trainset")
bst = None
for chunk_idx, train_df in enumerate(get_chunk_data(train_file,nrows=-1)):
    start_time = time.time()
    (bst,best_iteration) = lgb_modelfit_nocv(bst,param_dict,
                                             train_df,
                                             val_df,
                                             predictors,
                                             target,
                                             objective='binary',
                                             metrics='auc',
                                             early_stopping_rounds=30,
                                             verbose_eval=True,
                                             num_boost_round=1000,
                                             categorical_features=categorical)

    print('Chunk:{}, [{}]: model training time'.format(chunk_idx,time.time() - start_time))
    del train_df
    gc.collect()

Ver="v30"
date_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
data_tag = "%s_%s"%(Ver,date_str)

bst.save_model("lgb_val_{}.model".format(data_tag),best_iteration)

print("********* Predict the valset")
val = pd.DataFrame()
val['predicted'] = bst.predict(val_df[predictors],num_iteration=best_iteration)
#     if not debug:
#         print("writing...")
val.to_csv('val_pred.%s.csv'%(data_tag),index=False,float_format='%.9f')
print("done...")

# Feature names:
print('Feature names:', bst.feature_name())
# Feature importances:
print('Feature importances:', list(bst.feature_importance()))

feature_imp = pd.DataFrame(bst.feature_name(),list(bst.feature_importance()))
print(feature_imp)

#ax = lgb.plot_importance(bst, max_num_features=300)

#plt.savefig('test%d.png'%(fileno), dpi=600,bbox_inches="tight")
#plt.show()

print("*********** Predict with all data")
start_time = time.time()
(bst,best_iteration) = lgb_modelfit_nocv(bst,param_dict,
                                         val_df,
                                         val_df,
                                         predictors,
                                         target,
                                         objective='binary',
                                         metrics='auc',
                                         early_stopping_rounds=30,
                                         verbose_eval=True,
                                         num_boost_round=best_iteration,
                                         categorical_features=categorical)

print('[{}]: model training time'.format(time.time() - start_time))
del val_df
gc.collect()

bst.save_model("lgb_all_{}.model".format(data_tag),best_iteration)

print("Read test data")
test_df = pd.read_csv(test_file,engine='c',usecols=predictors+['click_id'],dtype=dtypes)
print("test.len={}".format(len(test_df)))

print("********** Predicting...")
sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
#     if not debug:
#         print("writing...")
sub.to_csv('sub_pred.%s.csv'%(data_tag),index=False,float_format='%.9f')
print("done...")


