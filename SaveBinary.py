#coding: utf-8

train_file="../newdata/train_v30.csv"
test_file="../newdata/test_v30.csv"
valid_file="../newdata/valid_v30.csv"


import lightgbm as lgb


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

print("Load train data")
dtrain = pd.read_csv(train_file,engine='c',usecols=predictors + [target], dtype=dtypes)
print("Load valid data")
dvalid = pd.read_csv(valid_file,engine='c',usecols=predictors+[target],dtype=dtypes)
#dtest = pd.read_csv(test_file,engine='c',usecols=predictors+[target],dtype=dtypes)

print("Make train dataset")
xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
print("Save train binary dataset")
xgtrain.save_binary("../newdata/train_v30.csv.bin")


print("Make valid dataset")
xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
print("Save valid binary dataset")
xgvalid.save_binary("../newdata/valid_v30.csv.bin")



