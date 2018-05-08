import sys

sys.path.insert(0, '../input/wordbatch-133/wordbatch/')
sys.path.insert(0, '../input/randomstate/randomstate/')
import wordbatch
from wordbatch.extractors import WordHash
from wordbatch.models import FM_FTRL
from wordbatch.data_utils import *
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

def df_add_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols)+'_count'] = counts[unqtags]

def df2csr(df, pick_hours=None):
    df.reset_index(drop=True, inplace=True)
    with timer("Adding counts"):
        df['click_time']= pd.to_datetime(df['click_time'])
        dt= df['click_time'].dt
        df['day'] = dt.day.astype('uint8')
        df['hour'] = dt.hour.astype('uint8')
        del(dt)
        df_add_counts(df, ['ip', 'day', 'hour'])
        df_add_counts(df, ['ip', 'app'])
        df_add_counts(df, ['ip', 'app', 'os'])
        df_add_counts(df, ['ip', 'device'])
        df_add_counts(df, ['app', 'channel'])

        # Add Features
        df_add_counts(df, ['ip', 'app', 'hour'])
        df_add_counts(df, ['ip', 'app', 'os','hour'])
        df_add_counts(df, ['ip', 'device','hour'])
        df_add_counts(df, ['app', 'channel','hour'])
        df_add_counts(df, ['ip', 'app', 'channel'])
        df_add_counts(df, ['ip', 'app', 'channel','hour'])

    #cpuStats()

    with timer("Adding next click times"):
        D= 2**26
        df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) \
                          + "_" + df['os'].astype(str)).apply(hash) % D
        click_buffer= np.full(D, 3000000000, dtype=np.uint32)
        df['epochtime']= df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks= []
        for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
            next_clicks.append(click_buffer[category]-time)
            click_buffer[category]= time
        del(click_buffer)
        df['next_click']= list(reversed(next_clicks))

    for fea in ['ip_day_hour_count','ip_app_count','ip_app_os_count','ip_device_count',
                'app_channel_count',
                'ip_app_hour_count','ip_app_os_hour_count','ip_device_hour_count',
                'app_channel_hour_count','ip_app_channel_count','ip_app_channel_hour_count',
                'next_click']:
        df[fea]= np.log2(1 + df[fea].values).astype(int)

    """
    with timer("Generating str_array"):
        str_array= ("I" + df['ip'].astype(str) \
                    + " A" + df['app'].astype(str) \
                    + " D" + df['device'].astype(str) \
                    + " O" + df['os'].astype(str) \
                    + " C" + df['channel'].astype(str) \
                    + " WD" + df['day'].astype(str) \
                    + " H" + df['hour'].astype(str) \
                    + " AXC" + df['app'].astype(str)+"_"+df['channel'].astype(str) \
                    + " OXC" + df['os'].astype(str)+"_"+df['channel'].astype(str) \
                    + " AXD" + df['app'].astype(str)+"_"+df['device'].astype(str) \
                    + " IXA" + df['ip'].astype(str)+"_"+df['app'].astype(str) \
                    + " AXO" + df['app'].astype(str)+"_"+df['os'].astype(str) \
                    + " IDHC" + df['ip_day_hour_count'].astype(str) \
                    + " IAC" + df['ip_app_count'].astype(str) \
                    + " AOC" + df['ip_app_os_count'].astype(str) \
                    + " IDC" + df['ip_device_count'].astype(str) \
                    + " AC" + df['app_channel_count'].astype(str) \
                    + " NC" + df['next_click'].astype(str)
                    ).values
    """
    #cpuStats()
    if 'is_attributed' in df.columns:
        df['click_id'] = df['is_attributed'].values
        df['weight'] = np.multiply([1.0 if x == 1 else 0.2 for x in df['is_attributed'].values],
                              df['hour'].apply(lambda x: 1.0 if x in pick_hours else 0.5))
        #df.drop('is_attributed')
    else:
        df['weight'] = 1
        #labels = []
        #weights = []
    return df #, labels, weights

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
idx = 0
for df_c in pd.read_csv('data/train.csv', engine='c', chunksize=batchsize,
                        #for df_c in pd.read_csv('../input/train.csv', engine='c', chunksize=batchsize,
                        sep=",", dtype=dtypes):
    #cpuStats()
    df = df2csr(df_c, pick_hours={4, 5, 10, 13, 14})
    df[features].to_csv('newdata/train_wb_{}.csv'.format(idx),index=False)
    idx += 1
    del(df)
    del(df_c)
    gc.collect()

idx = 0
for df_c in pd.read_csv('data/test.csv', engine='c', chunksize=batchsize,
                        #for df_c in pd.read_csv('../input/test.csv', engine='c', chunksize=batchsize,
                        sep=",", dtype=dtypes):
    df = df2csr(df_c)
    df[features].to_csv('newdata/test_wb_{}.csv'.format(idx),index=False)
    idx += 1
    del(df)
    del(df_c)
    gc.collect()

