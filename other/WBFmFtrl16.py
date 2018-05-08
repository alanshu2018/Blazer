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

def df2csr(wb, df, pick_hours=None):
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

batchsize = 20000000
D = 2 ** 20

dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
}

class WBFmFtrlModel(object):
    wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
                                                     "lowercase": False, "n_features": D,
                                                     "norm": None, "binary": True})
                         , minibatch_size=batchsize // 80, procs=8, freeze=True, timeout=1800, verbose=0)
    clf = FM_FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, alpha_fm=0.02, L2_fm=0.0, init_fm=0.01, weight_fm=1.0,
              D_fm=8, e_noise=0.0, iters=3, inv_link="sigmoid", e_clip=1.0, threads=4, use_avx=1, verbose=0)

    def __init__(self,train_files):
        self.train_files = train_files

    def predict(self,predict_file):
        p = None
        test_preds = []
        click_ids = []
        X = None
        for df_c in pd.read_csv(predict_file,engine='c',chunksize=batchsize,sep=","):
            str_array, labels, weights = df2csr(self.wb,df_c)
            click_ids+= df_c['click_id'].tolist()
            del(df_c)
            if p != None:
                test_preds += list(p.join())
                if X is not None:
                    del (X)
                    X = None
            gc.collect()
            X = self.wb.transform(str_array)
            del (str_array)
            p = ThreadWithReturnValue(target=predict_batch, args=(self.clf, X))
            p.start()

        if p != None:  test_preds += list(p.join())
        del(X)
        return click_ids, test_preds

    def train(self):
        p = None
        X = None
        rcount = 0
        for train_file in self.train_files:
            print("Train using file:{}".format(train_file))
            for df_c in pd.read_csv(train_file, engine='c', chunksize=batchsize,
                        #for df_c in pd.read_csv('../input/train.csv', engine='c', chunksize=batchsize,
                        sep=",", dtype=dtypes):
                rcount += len(df_c)
                #cpuStats()
                str_array, labels, weights= df2csr(self.wb, df_c, pick_hours={4, 5, 10, 13, 14})
                del(df_c)
                if p != None:
                    p.join()
                    if X is not None:
                        del(X)
                        X = None
                gc.collect()
                X= self.wb.transform(str_array)
                del(str_array)
                if rcount % (2 * batchsize) == 0:
                    if p != None:  p.join()
                    p = threading.Thread(target=evaluate_batch, args=(self.clf, X, labels, rcount))
                    p.start()
                print("Training", rcount, time.time() - start_time)
                cpuStats()
                if p != None:  p.join()
                p = threading.Thread(target=fit_batch, args=(self.clf, X, labels, weights))
                p.start()
                if p != None:  p.join()

                del(X)
                X = None

Ver = "16"
date_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
data_tag = "%s_%s"%(Ver,date_str)

print("===First CV to verify the AUC and ACC, and generate the predicted by CV")
train_file_idxs=[]
train_file_idxs.extend(range(6))
valid_file_idxs=[6,7,8,9]
test_preds = []
aucs = []
for idx, valid_idx in enumerate(valid_file_idxs):
    train_files = [ "newdata/train_wb_{}.csv".format(idx) for idx in train_file_idxs]
    valid_file = "newdata/train_wb_{}.csv".format(valid_idx)
    for file_idx in valid_file_idxs:
        if file_idx != valid_idx:
            train_files.append("newdata/train_wb_{}.csv".format(file_idx))

    print("VC train: idx={}".format(idx))
    print("Train_files={}".format(','.join(train_files)))
    print("Valid file={}".format(valid_file))

    model = WBFmFtrlModel(train_files=train_files)
    print("====Train Begin")
    model.train()
    print("====Predict")

    click_ids, preds = model.predict(valid_file)
    aucs.append(roc_auc_score(click_ids,preds))
    test_preds.extend(preds)

print("CV aucs={},mean={},running mean_auc={}".format(aucs,sum(aucs)/len(aucs),mean_auc))

print("Save CV predictions")
df_sub = pd.DataFrame({'predicted': test_preds})
df_sub.to_csv("cv_wordbatch_fm_ftrl_{}.csv".format(data_tag), index=False)
print("Done!")

print("====Train model using all the data")
train_files=["newdata/train_wb_{}.csv".format(i) for i in range(10)]
model = WBFmFtrlModel(train_files=train_files)
print("====Train Begin")
model.train()
print("====Train End")

print("====Generate submissions using model")
click_ids, test_preds = model.predict("newdata/test_wb_0.csv")
df_sub = pd.DataFrame({"click_id": click_ids, 'is_attributed': test_preds})
df_sub.to_csv("sub_wordbatch_fm_ftrl_{}.csv".format(data_tag), index=False)
print("Done!")
