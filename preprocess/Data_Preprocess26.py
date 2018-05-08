import sys

sys.path.insert(0, '../input/wordbatch-133/wordbatch/')
sys.path.insert(0, '../input/randomstate/randomstate/')
import threading
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
import numpy as np
import pandas as pd
import gc
from contextlib import contextmanager
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{name}] done in {time.time() - t0:.0f} s')

import os, psutil
def cpuStats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print('memory GB:', memoryUse)

start_time = time.time()

def do_next_Click( df,agg_suffix='nextClick', agg_type='float32'):

    print(">> \nExtracting {agg_suffix} time calculation features...\n")

    GROUP_BY_NEXT_CLICKS = [

        # V1
        # {'groupby': ['ip']},
        # {'groupby': ['ip', 'app']},
        # {'groupby': ['ip', 'channel']},
        # {'groupby': ['ip', 'os']},

        # V3
        {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
        {'groupby': ['ip', 'os', 'device']},
        {'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:

        # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)

        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df[all_features].groupby(spec[
                                                        'groupby']).click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)

        #predictors.append(new_feature)
        gc.collect()
    return (df)

def do_prev_Click( df,agg_suffix='prevClick', agg_type='float32'):

    print(">> \nExtracting {agg_suffix} time calculation features...\n")

    GROUP_BY_NEXT_CLICKS = [

        # V1
        # {'groupby': ['ip']},
        # {'groupby': ['ip', 'app']},
        {'groupby': ['ip', 'channel']},
        {'groupby': ['ip', 'os']},

        # V3
        #{'groupby': ['ip', 'app', 'device', 'os', 'channel']},
        #{'groupby': ['ip', 'os', 'device']},
        #{'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:

        # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)

        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        #print(">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df.click_time - df[all_features].groupby(spec[
                                                                        'groupby']).click_time.shift(+1) ).dt.seconds.astype(agg_type)

        #predictors.append(new_feature)
        gc.collect()
    return (df)

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



def df2csr(train_df, pick_hours=None):
    train_df.reset_index(drop=True, inplace=True)
    with timer("Adding counts"):
        train_df['click_time']= pd.to_datetime(train_df['click_time'])
        dt= train_df['click_time'].dt
        train_df['day'] = dt.day.astype('uint8')
        train_df['hour'] = dt.hour.astype('uint8')
        del(dt)

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


        train_df = do_prev_Click(train_df); gc.collect()
        train_df = do_next_Click(train_df); gc.collect()


    #cpuStats()
    if 'is_attributed' in train_df.columns:
        train_df['click_id'] = train_df['is_attributed'].values
        train_df['weight'] = np.multiply([1.0 if x == 1 else 0.2 for x in train_df['is_attributed'].values],
                              train_df['hour'].apply(lambda x: 1.0 if x in pick_hours else 0.5))
        #train_df.drop('is_attributed')
    else:
        train_df['weight'] = 1
        #labels = []
        #weights = []

    try:
        train_df.drop['click_time']
    except:
        pass

    return train_df #, labels, weights

batchsize = 20000000
dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
}

p = None
rcount = 0
features=['weight','click_id',
          'ip','app','device','os','channel','day','hour',
          'ip_day_hour_count','ip_app_count','ip_app_os_count','ip_device_count',
          'app_channel_count',
          'ip_app_hour_count','ip_app_os_hour_count','ip_device_hour_count',
          'app_channel_hour_count','ip_app_channel_count','ip_app_channel_hour_count',
          'next_click']

predictors = [
'app_cu_chl', 'ip', 'app', 'ip_cu_c', 'ip_da_chl_var_h', 'ip_app_os_var_h', 'device', 'app_d_h_co', #8
'app_chl_var_h', 'ip_d_os_c_app', 'app_chl_h_co', 'next_click', 'app_d_co', 'ip_d_co', 'app_os_var_da', #7
'ip_da_cu_h', 'channel', 'app_os_h_co', 'app_chl_co', 'ip_app_chl_var_h', 'ip_os_co', 'ip_app_co', #7
'app_os_co', 'ip_d_os_cu_app', 'app_d_var_h', 'ip_chl_var_h', 'hour', 'ip_app_cu_os', 'ip_d_h_co', #7
'ip_app_chl_var_da', 'app_os_mean_h', 'ip_cu_app', 'ip_os_h_co', 'os','next_click_shift' #6
]

idx = 0
for df_c in pd.read_csv('data/train.csv', engine='c', chunksize=batchsize,
                        #for df_c in pd.read_csv('../input/train.csv', engine='c', chunksize=batchsize,
                        sep=",", dtype=dtypes):
    #cpuStats()
    df = df2csr(df_c, pick_hours={4, 5, 10, 13, 14})
    df.to_csv('newdata/train_v26_{}.csv'.format(idx),index=False)
    idx += 1
    del(df)
    del(df_c)
    gc.collect()

idx = 0
for df_c in pd.read_csv('data/test.csv', engine='c', chunksize=batchsize,
                        #for df_c in pd.read_csv('../input/test.csv', engine='c', chunksize=batchsize,
                        sep=",", dtype=dtypes):
    df = df2csr(df_c)
    df.to_csv('newdata/test_v26_{}.csv'.format(idx),index=False)
    idx += 1
    del(df)
    del(df_c)
    gc.collect()

