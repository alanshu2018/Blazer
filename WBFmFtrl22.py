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
import gc
import config

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

mean_auc= 0

def fit_batch(clf, X, y, w):  clf.partial_fit(X, y, sample_weight=w)

def predict_batch(clf, X):  return clf.predict(X)

def evaluate_batch(clf, X, y, rcount):
    auc= roc_auc_score(y, predict_batch(clf, X))
    global mean_auc
    if mean_auc==0:
        mean_auc= auc
    else: mean_auc= 0.2*(mean_auc*4 + auc)
    print(rcount, "ROC AUC:", auc, "Running Mean:", mean_auc)
    return auc

def df_add_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols)+'_count'] = counts[unqtags]

def df2csr(data):
    data_shape = data.shape

    print("data_shape={}".format(data_shape))
    str_array = np.apply_along_axis(lambda row: " ".join(["{}{}".format(chr(65+i),row[i]) for i in range(data_shape[1])]), 1, data)
    del(data)
    gc.collect()
    print("str_array_shape={}".format(str_array.shape))
    return str_array

def df2csr1(wb, df, pick_hours=None):
    with timer("Generating str_array"):
        # Add six features
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
                    + " NC" + df['next_click'].astype(str) \
                    + " IAH" + df['ip_app_hour_count'].astype(str) \
                    + " IAOH" + df['ip_app_os_hour_count'].astype(str) \
                    + " IDH" + df['ip_device_hour_count'].astype(str) \
                    + " ACHH" + df['app_channel_hour_count'].astype(str) \
                    + " IACH" + df['ip_app_channel_count'].astype(str) \
                    + " IACH" + df['ip_app_channel_hour_count'].astype(str)
                    ).values
    #cpuStats()
    weights = df['weight'].values
    labels = df['click_id'].values
    return str_array, labels, weights

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self):
        threading.Thread.join(self)
        return self._return

batchsize = 10000000
D = 2 ** 20

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

predictors = [
'app_cu_chl', 'ip', 'app', 'ip_cu_c', 'ip_da_chl_var_h', 'ip_app_os_var_h', 'device', 'app_d_h_co', #8
'app_chl_var_h', 'ip_d_os_c_app', 'app_chl_h_co', 'next_click', 'app_d_co', 'ip_d_co', 'app_os_var_da', #7
'ip_da_cu_h', 'channel', 'app_os_h_co', 'app_chl_co', 'ip_app_chl_var_h', 'ip_os_co', 'ip_app_co', #7
'app_os_co', 'ip_d_os_cu_app', 'app_d_var_h', 'ip_chl_var_h', 'hour', 'ip_app_cu_os', 'ip_d_h_co', #7
'ip_app_chl_var_da', 'app_os_mean_h', 'ip_cu_app', 'ip_os_h_co', 'os','next_click_shift' #6
]


class WBFmFtrlModel(object):
    wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
                                                     "lowercase": False, "n_features": D,
                                                     "norm": None, "binary": True})
                         , minibatch_size=batchsize // 80, procs=8, freeze=True, timeout=1800, verbose=0)
    #clf = FM_FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, alpha_fm=0.02, L2_fm=0.0, init_fm=0.01, weight_fm=1.0,
    #          D_fm=8, e_noise=0.0, iters=3, inv_link="sigmoid", e_clip=1.0, threads=4, use_avx=1, verbose=0)

    def __init__(self,train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.clf = None

    def create_clf(self):
        if self.clf is not None:
            del(self.clf)
            gc.collect()
        self.clf = FM_FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, alpha_fm=0.02, L2_fm=0.0, init_fm=0.01, weight_fm=1.0,
                      D_fm=16, e_noise=0.0, iters=5, inv_link="sigmoid", e_clip=1.0, threads=4, use_avx=1, verbose=0)

    def predict(self,predict_file):
        p = None
        test_preds = []
        click_ids = []
        X = None
        for df_c in pd.read_csv(predict_file,engine='c',chunksize=batchsize,sep=",",usecols=predictors+["click_id","weight"]):
            str_array = df2csr(df_c[predictors].values)
            labels = df_c["click_id"].values
            weights = df_c["weight"].values
            click_ids+= df_c['click_id'].tolist()
            del(df_c)
            if p != None:
                test_preds += list(p.join())
                if X is not None:
                    del (X)
                    gc.collect()
            X = self.wb.transform(str_array)
            del (str_array)
            p = ThreadWithReturnValue(target=predict_batch, args=(self.clf, X))
            p.start()

        if p != None:  test_preds += list(p.join())
        if X is not None:
            del(X)
            gc.collect()
        return click_ids, test_preds

    def read_data_file(self,train_file,skip_rows, nrows):
        if skip_rows>0:
            skip_rows = range(1,skip_rows)
        else:
            skip_rows = None
        df_c = pd.read_csv(train_file,skiprows=skip_rows, nrows = nrows,engine="c",dtype=dtypes,usecols=predictors+["weight","click_id"])
        str_array = df2csr((df_c[predictors].values))
        X= self.wb.transform(str_array)
        labels = df_c["click_id"].values
        weights = df_c["weight"].values
        del(str_array)
        del(df_c)
        gc.collect()
        return X, labels, weights

    def predict_data(self, X, labels, weights):
        return predict_batch(self.clf, X)

    def train_all(self):
        p = None
        X = None
        rcount = 0
        if True:
            start_time = time.time()

            self.create_clf()

            print("Train using file:{}".format(self.train_file))
            print("Pretrain the model")
            start = 24903889
            start_loops = int(start/batchsize)
            pos = 0
            for i in range(start_loops+1):
                if p != None:
                    p.join()
                    if X is not None:
                        del(X)
                        X = None
                        del(labels)
                        del(weights)
                        gc.collect()
                nrows = batchsize
                if pos + batchsize > start:
                    nrows = start - pos + 1

                if nrows <=1:
                    break

                print("Pretrain: pos={}, nrows={}".format(pos, nrows))
                if pos <=0:
                    X, labels, weights = self.read_data_file(self.train_file,0,nrows)
                    pos += nrows
                else:
                    skip = pos - batchsize
                    X, labels, weights = self.read_data_file(self.train_file,skip,nrows)
                    pos += nrows
                p = threading.Thread(target=fit_batch, args=(self.clf, X, labels, weights))
                p.start()

            rcount += start
            print("Training", rcount, time.time() - start_time)
            # First train
            tv = [batchsize,batchsize*2,batchsize*3,batchsize*4]
            for idx, pos in enumerate(tv):
                skip = start + pos - batchsize
                if p != None:
                    p.join()
                    if X is not None:
                        del(X)
                        X = None
                        del(labels)
                        del(weights)
                        gc.collect()
                X, labels, weights = self.read_data_file(self.train_file,skip,batchsize)
                rcount += batchsize
                if idx >= 1:
                    if p != None:  p.join()
                    p = threading.Thread(target=evaluate_batch, args=(self.clf, X, labels, rcount))
                    p.start()
                if p != None:  p.join()
                print("Training", rcount, time.time() - start_time)
                p = threading.Thread(target=fit_batch, args=(self.clf, X, labels, weights))
                p.start()

            if p != None:  p.join()
            if X is not None:
                del(X)
                X = None
                del(labels)
                del(weights)
                gc.collect()

    def train_cv(self):
        p = None
        X = None
        rcount = 0
        if True:
            start_time = time.time()

            train_valids = [
                [batchsize,batchsize*2,batchsize*3,batchsize*4],
                [batchsize*2,batchsize*3,batchsize*4,batchsize],
                [batchsize,batchsize*3,batchsize*4,batchsize*2],
                [batchsize,batchsize*2,batchsize*4,batchsize*3],
            ]

            all_cv_preds = np.zeros(shape=(4*batchsize,),dtype=np.float16)
            for tv in train_valids:
                print("Train_CV: tv={}".format(tv))
                self.create_clf()

                print("Train using file:{}".format(self.train_file))
                print("Pretrain the model")
                start = 24903889
                start_loops = int(start/batchsize)
                pos = 0
                for i in range(start_loops+1):
                    if p != None:
                        p.join()
                        if X is not None:
                            del(X)
                            X = None
                            del(labels)
                            del(weights)
                            gc.collect()
                    nrows = batchsize
                    if pos + batchsize > start:
                        nrows = start - pos + 1

                    if nrows <=1:
                        break

                    print("Pretrain: pos={}, nrows={}".format(pos, nrows))
                    if pos <=0:
                        X, labels, weights = self.read_data_file(self.train_file,0,nrows)
                        pos += nrows
                    else:
                        skip = pos - batchsize
                        X, labels, weights = self.read_data_file(self.train_file,skip,nrows)
                        pos += nrows
                    p = threading.Thread(target=fit_batch, args=(self.clf, X, labels, weights))
                    p.start()

                rcount += start
                print("Training", rcount, time.time() - start_time)
                # First train
                for idx, pos in enumerate(tv[:3]):
                    skip = start + pos - batchsize
                    if p != None:
                        p.join()
                        if X is not None:
                            del(X)
                            X = None
                            del(labels)
                            del(weights)
                            gc.collect()
                    X, labels, weights = self.read_data_file(self.train_file,skip,batchsize)
                    rcount += batchsize
                    if idx % 2 == 1:
                        if p != None:  p.join()
                        p = threading.Thread(target=evaluate_batch, args=(self.clf, X, labels, rcount))
                        p.start()
                    if p != None:  p.join()
                    print("Training", rcount, time.time() - start_time)
                    p = threading.Thread(target=fit_batch, args=(self.clf, X, labels, weights))
                    p.start()

                if p != None:  p.join()
                if X is not None:
                    del(X)
                    X = None
                    del(labels)
                    del(weights)
                    gc.collect()

                print("Predict for the validation data")
                pos = tv[3]
                skip = start + pos -batchsize
                X, labels, weights = self.read_data_file(self.train_file,skip,batchsize)
                pred = predict_batch(self.clf, X)
                all_cv_preds[pos-batchsize:pos] = np.reshape(pred,(batchsize,))
                if X is not None:
                    del(X)
                    X = None
                    del(labels)
                    del(weights)
                    gc.collect()

            # Save cv result data
            fname = "%s/cv_pred_%s_%s.csv"%(config.OUTPUT_DIR, "fmftrl",Ver)
            print("Save cv predictions:{}".format(fname))
            df = pd.DataFrame({"predicted": all_cv_preds})
            df.to_csv(fname, index=False, columns=["predicted"])


Ver = "22"
date_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
data_tag = "%s_%s"%(Ver,date_str)

print("===First CV to verify the AUC and ACC, and generate the predicted by CV")
print("====Train model using all the data")
train_file="../newdata/train_v22.csv" #["newdata/train_wb_{}.csv".format(i) for i in range(10)]
test_file="../newdata/test_v22.csv" #["newdata/train_wb_{}.csv".format(i) for i in range(10)]
model = WBFmFtrlModel(train_file, test_file)
print("====Train CV Begin")
#model.train_cv()
print("====Train CV End")

print("====Train ALL Begin")
model.train_all()
print("====Train ALL End")

print("====Generate submissions using model")
click_ids, test_preds = model.predict(test_file)
df_sub = pd.DataFrame({"click_id": click_ids, 'is_attributed': test_preds})
fname = "%s/sub_pred_fmftrl_%s.csv"%(config.SUBM_DIR, data_tag)
df_sub.to_csv(fname, index=False)
print("Done!")
