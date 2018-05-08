#coding:utf-8


import pandas as pd
import numpy as np

filename_v22 = "../newdata/train_v22.csv"
filename_v28 = "../newdata/train_v22.csv"

filename = filename_v28

nrows = 20000000
#nrows = 20000

predictors_v22 = [
     'weight','click_id',
     'ip','app','device','channel','os','hour',
     'app_cu_chl', 'ip_cu_c', 'ip_da_chl_var_h', 'ip_app_os_var_h', 'app_d_h_co', #8
     'app_chl_var_h', 'ip_d_os_c_app', 'app_chl_h_co', 'next_click', 'app_d_co', 'ip_d_co', 'app_os_var_da', #7
     'ip_da_cu_h', 'app_os_h_co', 'app_chl_co', 'ip_app_chl_var_h', 'ip_os_co', 'ip_app_co', #7
     'app_os_co', 'ip_d_os_cu_app', 'app_d_var_h', 'ip_chl_var_h', 'ip_app_cu_os', 'ip_d_h_co', #7
     'ip_app_chl_var_da', 'app_os_mean_h', 'ip_cu_app', 'ip_os_h_co', 'next_click_shift' #6
    ]
predictors_v28 = [
    'app_cu_chl', 'ip', 'app', 'ip_cu_c', 'ip_da_chl_var_h', 'ip_app_os_var_h', 'device', 'app_d_h_co', #8
    'app_chl_var_h', 'ip_d_os_c_app', 'app_chl_h_co', 'app_d_co', 'ip_d_co', 'app_os_var_da', #7
    'ip_da_cu_h', 'channel', 'app_os_h_co', 'app_chl_co', 'ip_app_chl_var_h', 'ip_os_co', 'ip_app_co', #7
    'app_os_co', 'ip_d_os_cu_app', 'app_d_var_h', 'ip_chl_var_h', 'hour', 'ip_app_cu_os', 'ip_d_h_co', #7
    'ip_app_chl_var_da', 'app_os_mean_h', 'ip_cu_app', 'ip_os_h_co', 'os', #6
    'ip_channel_prevClick','ip_os_prevClick','ip_app_device_os_channel_nextClick',
    'ip_os_device_nextClick','ip_os_device_app_nextClick',
]

predictors = predictors_v28

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

df = pd.read_csv(filename,nrows=nrows,dtype=dtypes)
print("num_predictors={}".format(len(predictors)))
print("\nMean")
print("{}".format("\t".join(predictors)))
print("{}".format(df.mean()[predictors].values.tolist()))
print("\nMax")
print("{}".format("\t".join(predictors)))
print("{}".format(df.max()[predictors].values.tolist()))
print("\nMin")
print("{}".format("\t".join(predictors)))
print("{}".format(df.min()[predictors].values.tolist()))
