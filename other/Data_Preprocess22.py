"""
A non-blending lightGBM model that incorporates portions and ideas from various public kernels
This kernel gives LB: 0.977 when the parameter 'debug' below is set to 0 but this implementation requires a machine with ~32 GB of memory
"""

import pandas as pd
import time
import numpy as np
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc
import os
import sys

Ver = "22"

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

print("Read Training Data")

train_file = "newdata/train_v20.csv"
train_df = pd.read_csv(train_file, dtype=dtypes,skiprows=(1,24903891),usecols=['ip','app','device','os', 'channel', 'day','hour', 'click_id','weight','next_click'])
len_train = len(train_df)
print("train.len={}".format(len_train))

#sys.exit(-1)

print("Read Testing Data")
test_file = "newdata/test_v20.csv"
test_df = pd.read_csv(test_file, dtype=dtypes,usecols=['ip','app','device','os', 'channel', 'day','hour', 'click_id','weight','next_click'])
train_df1 = pd.concat([train_df,test_df],axis=0)
del(train_df)
del(test_df)
train_df = train_df1
len_all = len(train_df)
print("total.len={}".format(len_all))
gc.collect()
print("Begin Data_Preprocessing")

def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

print('Extracting aggregation features...')
#2
train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'ip_d_os_c_app', show_max=True ); gc.collect()
train_df = do_cumcount( train_df, ['ip'], 'os', 'ip_c_os', show_max=True ); gc.collect()
#7
train_df = do_countuniq( train_df, ['ip'], 'channel', 'ip_cu_c', 'uint8', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'day'], 'hour', 'ip_da_cu_h', 'uint8', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'app', 'ip_cu_app', 'uint8', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'app'], 'os', 'ip_app_cu_os', 'uint8', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'device', 'ip_cu_d', 'uint8', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['app'], 'channel', 'app_cu_chl', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'ip_d_os_cu_app', show_max=True ); gc.collect()

#11
train_df = do_count( train_df, ['ip', 'day'], 'ip_da_co', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'app'], 'ip_app_co', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'app', 'os'], 'ip_app_os_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'device'], 'ip_d_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['app', 'channel'], 'app_chl_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'channel'], 'ip_ch_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip','app', 'channel'], 'ip_app_chl_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['app','device'], 'app_d_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['app','os'], 'app_os_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip','os'], 'ip_os_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip','device','os','app'], 'ip_d_os_co', 'uint16', show_max=True ); gc.collect()

#10
train_df = do_count( train_df, ['ip', 'app','hour'], 'ip_app_h_co', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'app', 'os','hour'], 'ip_app_os_h_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'device','hour'], 'ip_d_h_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['app', 'channel','hour'], 'app_chl_h_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'channel','hour'], 'ip_ch_h_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip','app', 'channel','hour'], 'ip_app_chl_h_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['app','device','hour'], 'app_d_h_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['app','os','hour'], 'app_os_h_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip','os','hour'], 'ip_os_h_co', 'uint16', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip','device','os','app','hour'], 'ip_d_os_h_co', 'uint16', show_max=True ); gc.collect()

#16
train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour', 'ip_da_chl_var_h', show_max=True ); gc.collect()
train_df = do_var( train_df, ['ip', 'channel'], 'hour', 'ip_chl_var_h', show_max=True ); gc.collect()
train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var_h', show_max=True ); gc.collect()
train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day', 'ip_app_chl_var_da', show_max=True ); gc.collect()
train_df = do_var( train_df, ['ip', 'app'], 'hour', 'ip_app_chl_var_h', show_max=True ); gc.collect()
train_df = do_var( train_df, ['app','os'], 'hour', 'app_os_var_da', show_max=True ); gc.collect()
train_df = do_var( train_df, ['app','device'], 'hour', 'app_d_var_h', show_max=True ); gc.collect()
train_df = do_var( train_df, ['app','channel'], 'hour', 'app_chl_var_h', show_max=True ); gc.collect()
train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_chl_mean_h', show_max=True ); gc.collect()
train_df = do_mean( train_df, ['ip', 'channel'], 'hour', 'ip_chl_mean_h', show_max=True ); gc.collect()
train_df = do_mean( train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_mean_h', show_max=True ); gc.collect()
train_df = do_mean( train_df, ['ip', 'app'], 'hour', 'ip_app_mean_h', show_max=True ); gc.collect()
train_df = do_mean( train_df, ['app','os'], 'hour', 'app_os_mean_h', show_max=True ); gc.collect()
train_df = do_mean( train_df, ['app','device'], 'hour', 'app_mean_var_h', show_max=True ); gc.collect()
train_df = do_mean( train_df, ['app','channel'], 'hour', 'app_chl_mean_h', show_max=True ); gc.collect()

print('Doing nextClick...')
predictors=[]
new_feature = 'next_click'
QQ = train_df[new_feature].values
train_df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values
predictors.append(new_feature)
predictors.append(new_feature+'_shift')

print("vars and data type: ")
train_df.info()

target = 'is_attributed'
predictors.extend(['app','device','os', 'channel', 'hour', 'day',
                   'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                   'ip_app_os_count', 'ip_app_os_var',
                   'ip_app_channel_var_day','ip_app_channel_mean_hour',
                   'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])
categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
print('predictors',predictors)

test_df = train_df[len_train:]
train_df = train_df[:len_train]

print("train size: ", len(train_df))
print("test size : ", len(test_df))

train_df.to_csv('newdata/train_v{}.csv'.format(Ver),index=False)
test_df.to_csv('newdata/test_v{}.csv'.format(Ver),index=False)
