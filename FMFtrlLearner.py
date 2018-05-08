#coding:utf-8
#coding:utf-8

import numpy as np
from Learner import BaseLearner
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval


D = 2 ** 20
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

def fit_batch(clf, X, y, w):  clf.partial_fit(X, y, sample_weight=w)

def predict_batch(clf, X):  return clf.predict(X)

mean_auc = 0.0
def evaluate_batch(clf, X, y, rcount):
    auc= roc_auc_score(y, predict_batch(clf, X))
    global mean_auc
    if mean_auc==0:
        mean_auc= auc
    else: mean_auc= 0.2*(mean_auc*4 + auc)
    print(rcount, "ROC AUC:", auc, "Running Mean:", mean_auc)
    return auc

def df2csr(data):
    data_shape = data.shape

    print("data_shape={}".format(data_shape))
    str_array = np.apply_along_axis(lambda row: " ".join(["{}{}".format(chr(65+i),row[i]) for i in range(data_shape[1])]), 1, data)
    del(data)
    gc.collect()
    print("str_array_shape={}".format(str_array.shape))
    return str_array

class FMFtrlModel(object):
    wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
                                                     "lowercase": False, "n_features": D,
                                                     "norm": None, "binary": True})
                         , minibatch_size=batchsize // 80, procs=8, freeze=True, timeout=1800, verbose=0)

    def __init__(self,config):
        self.config = config
        self._build()

    def _build(self):
        D_fm = self.config['D_fm']
        iters = self.config['iters']
        e_clip = self.config['e_clip']
        alpha_fm = self.config['alpha_fm']
        weight_fm = self.config['weight_fm']
        threads = 8

        clf = FM_FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, alpha_fm=alpha_fm, L2_fm=0.0, init_fm=0.01, weight_fm=weight_fm,
              D_fm=D_fm, e_noise=0.0, iters=iters, inv_link="sigmoid", e_clip=e_clip, threads=threads, use_avx=1, verbose=0)
        self.model = clf

    def fit(self,data, y, validate=True, weight=None):
        total_data = len(data)
        p = None
        X = None
        rcount = 0
        start_time = time.time()
        #cpuStats()
        step = 200000
        epochs = int(total_data/step)+1
        for epoch in range(epochs):
            start = epoch * step
            end = start + step
            if start >= total_data:
                break
            if end > total_data:
                end = total_data

            str_array = df2csr(data[start:end])
            labels = y[start:end]
            if weight is not None:
                W = weight[start:end]
            else:
                W = None
            if p != None:
                p.join()
            if X is not None:
                del(X)
                gc.collect()
            X= self.wb.transform(str_array)
            del(str_array)
            gc.collect()
            rcount += step
            if rcount % (2 * step) == 0:
                if p != None:  p.join()
                p = threading.Thread(target=evaluate_batch, args=(self.model, X, labels, rcount))
                p.start()
            print("Training", rcount, time.time() - start_time)
            if p != None:  p.join()
            p = threading.Thread(target=fit_batch, args=(self.model, X, labels, W))
            p.start()
        if p != None:  p.join()
        del(X)
        gc.collect()

    def predict(self,X_train,weight=None):
        p = None
        test_preds = []
        click_ids = []
        str_array  = df2csr(X_train)
        del(X_train)
        gc.collect()
        X = self.wb.transform(str_array)
        del(str_array)
        gc.collect()
        p = ThreadWithReturnValue(target=predict_batch, args=(self.model, X))
        p.start()

        if p != None:  test_preds += list(p.join())
        del(X)
        gc.collect()
        return test_preds

## regression with Keras' deep neural network
class FMFtrlLearner(BaseLearner):
    name = "reg_fmftrl"

    param_space = {
        'D_fm':hp.uniform("D_fm",6,64),
        'iters':hp.choice("iters",[3,5,9,12]),
        'e_clip':hp.loguniform("e_clip",np.log(0.01),np.log(10.0)),
        'alpha_fm':hp.loguniform("alpha_fm",np.log(0.01),np.log(1.0)),
        'weight_fm':hp.loguniform("weight_fm",np.log(0.01),np.log(1.0)),
    }

    def create_model(self,params):
        self.params = params

        return FMFtrlModel(params)

