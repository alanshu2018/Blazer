#coding: utf-8

import pandas as pd
import numpy as np

train_file = "../newdata/train_v22.csv"

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
}

import lightgbm as lgb
import gc
from sklearn.model_selection import train_test_split
predictors=[
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
 ]
categorical = [
    'ip','app','device','os','channel','day','hour',
]

weight_predictor = "weight"
target_predictor = "click_id"

df = pd.read_csv(train_file,skiprows = range(1,44903891),dtype=dtypes)
y = df[target_predictor].values
X = df[predictors].values
weights = df[weight_predictor].values

total = len(X)
rand_idxs = range(total)
np.random.shuffle(rand_idxs)
train_idxs = rand_idxs[:-1000000]
test_idxs = rand_idxs[-1000000:]

X_train,y_train,W_train = X[train_idxs],y[train_idxs],weights[train_idxs]
X_test,y_test,W_test = X[test_idxs],y[test_idxs],weights[test_idxs]

dvalid = lgb.Dataset(X_test, label=y_test,
             feature_name=predictors,
             weight=W_test,
             )

dtrain = lgb.Dataset(X_train, label=y_train,
              feature_name=predictors,
              weight=W_train,
              )

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
    'device': 'gpu',
    'nthread': 8,
    'verbose': 1,
    'is_unbalance': True
    #'scale_pos_weight':99
}

model = lgb.train(params,
       dtrain,
       valid_sets=[dvalid,dtrain],
       valid_names=['valid','train'],
       num_boost_round=900,
       early_stopping_rounds=50,
       categorical_feature=categorical,
       verbose_eval=5,
       init_model= None,
       feval=None)
step = model.best_iteration
auc = model.best_score['valid']['auc']

# Nick's Feature Importance Plot
import matplotlib.pyplot as plt
lgb_model = model
f, ax = plt.subplots(figsize=[7,10])
lgb.plot_importance(lgb_model, ax=ax, max_num_features=len(predictors))
plt.title("Light GBM Feature Importance")
plt.savefig('feature_import_1.png')

# Feature names:
print('Feature names:', lgb_model.feature_name())
# Feature importances:
print('Feature importances:', list(lgb_model.feature_importance()))

feature_imp = pd.DataFrame(lgb_model.feature_name(),list(lgb_model.feature_importance()))
print(feature_imp)

