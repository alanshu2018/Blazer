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

from DataPiper import *

if sys.version_info.major == 3:
	import pickle as pkl
else:
	import cPickle as pkl

Ver = "26"

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
D = 2 ** 21

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

predictors = [
'app_cu_chl', 'ip', 'app', 'ip_cu_c', 'ip_da_chl_var_h', 'ip_app_os_var_h', 'device', 'app_d_h_co', #8
'app_chl_var_h', 'ip_d_os_c_app', 'app_chl_h_co', 'app_d_co', 'ip_d_co', 'app_os_var_da', #7
'ip_da_cu_h', 'channel', 'app_os_h_co', 'app_chl_co', 'ip_app_chl_var_h', 'ip_os_co', 'ip_app_co', #7
'app_os_co', 'ip_d_os_cu_app', 'app_d_var_h', 'ip_chl_var_h', 'hour', 'ip_app_cu_os', 'ip_d_h_co', #7
'ip_app_chl_var_da', 'app_os_mean_h', 'ip_cu_app', 'ip_os_h_co', 'os', #6
'ip_channel_prevClick','ip_os_prevClick','ip_app_device_os_channel_nextClick',
'ip_os_device_nextClick','ip_os_device_app_nextClick',
#'next_click','next_click_shift',
]

predictors1 = [
    'ip','app','device','os','channel','click_time','day','hour','ip_d_os_c_app','ip_c_os',
    'ip_cu_c','ip_da_cu_h','ip_cu_app','ip_app_cu_os','ip_cu_d','app_cu_chl','ip_d_os_cu_app',
    'ip_da_co','ip_app_co','ip_app_os_co','ip_d_co','app_chl_co','ip_ch_co','ip_app_chl_co',
    'app_d_co','app_os_co','ip_os_co','ip_d_os_co','ip_app_h_co','ip_app_os_h_co','ip_d_h_co',
    'app_chl_h_co','ip_ch_h_co','ip_app_chl_h_co','app_d_h_co','app_os_h_co','ip_os_h_co',
    'ip_d_os_h_co','ip_da_chl_var_h','ip_chl_var_h','ip_app_os_var_h','ip_app_chl_var_da',
    'ip_app_chl_var_h','app_os_var_da','app_d_var_h','app_chl_var_h','ip_app_chl_mean_h',
    'ip_chl_mean_h','ip_app_os_mean_h','ip_app_mean_h','app_os_mean_h','app_mean_var_h',
    'app_chl_mean_h','ip_channel_prevClick','ip_os_prevClick','ip_app_device_os_channel_nextClick',
    'ip_os_device_nextClick','ip_os_device_app_nextClick',
]


from utils import dist_utils, logging_utils, pkl_utils, time_utils
logname="fmftrl"
logger = logging_utils._get_logger(config.LOG_DIR, logname)
class WBFmFtrlModel(object):
    wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
                                                     "lowercase": False, "n_features": D,
                                                     "norm": None, "binary": True})
                         , minibatch_size=batchsize // 80, procs=8, freeze=True, timeout=1800, verbose=0)
    #clf = FM_FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, alpha_fm=0.02, L2_fm=0.0, init_fm=0.01, weight_fm=1.0,
    #          D_fm=8, e_noise=0.0, iters=3, inv_link="sigmoid", e_clip=1.0, threads=4, use_avx=1, verbose=0)

    def __init__(self,pretrain_files,train_file, test_file):
        self.pretrain_files = pretrain_files
        self.train_file = train_file
        self.test_file = test_file
        self.clf = None
        self.pretrain_model_fn = "wb_fmftrl_v26_pretrain.model"

    def create_clf(self):
        if self.clf is not None:
            del(self.clf)
            gc.collect()
        self.clf = FM_FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, alpha_fm=0.02, L2_fm=0.0, init_fm=0.01, weight_fm=1.0,
                      D_fm=16, e_noise=0.0, iters=5, inv_link="sigmoid", e_clip=1.0, threads=4, use_avx=1, verbose=0)

    def get_data(self, loader, fold= -1, chunk_size=10000000, file_size=40000000):
        if fold > 0:
            size_per_fold = int(file_size/fold)
        else:
            size_per_fold = chunk_size

        for (idx, df) in loader.get_chunk_data():
            data = df[predictors].values
            labels = df['click_id'].values
            weights = df['weight'].values
            if fold == -1:
                fold_num = -1
            else:
                fold_num = int(idx / size_per_fold)
            del(df)
            gc.collect()

            str_array = df2csr(data)
            X = self.wb.transform(str_array)
            del(str_array)
            del(data)
            gc.collect()
            yield (idx, fold_num, X, labels, weights)

    def do_thread_execute(self,target,clf, X, labels=None, weights=None,do_free=True):
        #str_array = df2csr(data)
        #gc.collect()
        #X = self.wb.transform(str_array)
        if labels is not None:
            args = (clf, X, labels, weights)
        else:
            args = (clf, X)
        p = ThreadWithReturnValue(target=target,args =args)
        p.start()
        ret = p.join()
        if do_free:
            del(X)
            if labels is not None:
                del(labels)
            if weights is not None:
                del(weights)
        gc.collect()

        return ret

    def predict(self,predict_file):
        test_preds = []
        click_ids = []
        test_loader = DataPiper(predict_file,logger)
        for (idx, fold_num, X, labels, weights) in self.get_data(test_loader):
            click_ids+= labels.tolist()
            test_preds += list(self.do_thread_execute(predict_batch,self.clf,X))

        return click_ids, test_preds

    def predict_data(self, X, labels, weights):
        return predict_batch(self.clf, X)

    def pretrain(self):
        p = None
        X = None
        rcount = 0

        start_time = time.time()

        self.create_clf()

        if not os.path.exists(self.pretrain_model_fn):
            print("Pretrain the model")
            for pretrain_file in self.pretrain_files:
                print("Pretrain using file:{}".format(pretrain_file))
                loader = DataPiper(pretrain_file,logger)
                for (idx, fold_num, X, labels, weights) in self.get_data(loader):
                    self.do_thread_execute(fit_batch,self.clf,X,labels,weights)

            with open(self.pretrain_model_fn,"wb") as f:
                params = self.clf.__getstate__() #self.create_clf()
                pkl.dump(params,f)
            #self.clf.pickle_model(self.pretrain_model_fn)
        else:
            with open(self.pretrain_model_fn,"rb") as f:
                params = pkl.load(f)
                self.clf.__setstate__(params)
            #self.clf.unpickle_model(self.pretrain_model_fn)

    def train_all(self):
        p = None
        X = None
        rcount = 0

        start_time = time.time()

        self.create_clf()

        print("Pretrain the model")
        self.pretrain()
        """
        for pretrain_file in self.pretrain_files:
            print("Pretrain using file:{}".format(pretrain_file))
            loader = DataPiper(pretrain_file,logger)
            for (idx, fold_num, X, labels, weights) in self.get_data(loader):
                self.do_thread_execute(fit_batch,self.clf,X,labels,weights)
        """

        print("Train with file={}".format(self.train_file))
        rcount = 0
        loader = DataPiper(self.train_file,logger)
        loops = 0
        for (idx, fold_num, X, labels, weights) in self.get_data(loader):
            if loops % 2 == 0:
                self.do_thread_execute(evaluate_batch,self.clf,X,labels,weights, do_free=False)
            loops += 1
            rcount += len(labels)

            print("Training", rcount, time.time() - start_time)
            self.do_thread_execute(fit_batch,self.clf,X,labels,weights)

    def train_cv(self):
        start_time = time.time()

        nfold = 4
        train_preds = []
        auc_cv = [0.0 for _ in range(nfold)]
        for fold in range(nfold):
            self.create_clf()
            print("Pretrain models")
            self.pretrain()
            """
            for pretrain_file in self.pretrain_files:
                print("Pretrain using file:{}".format(pretrain_file))
                loader = DataPiper(pretrain_file,logger)
                for (idx, fold_num, X, labels, weights) in self.get_data(loader):
                    self.do_thread_execute(fit_batch,self.clf,X,labels,weights)
            """
            print("Train with file={}".format(self.train_file))
            file_size = 40000000
            all_cv_preds = np.zeros(shape=(file_size,),dtype=np.float32)
            loader = DataPiper(self.train_file,logger)
            valid_datas = []
            loops = 0
            rcount = 0
            for (idx, fold_num, X, labels, weights) in self.get_data(loader,fold=nfold,file_size=file_size):
                print("fold_num={},fold={},nfold={}".format(fold_num,fold,nfold))
                if fold_num == fold:
                    valid_datas.append((idx,fold_num,X,labels,weights))
                    print("Add valid_datas:len={}".format(len(valid_datas)))
                    continue

                loops += 1
                rcount += len(labels)
                if loops % 2 == 0:
                    self.do_thread_execute(evaluate_batch,self.clf,X,labels,weights,do_free=False)

                print("Training", rcount, time.time() - start_time)
                self.do_thread_execute(fit_batch,self.clf,X,labels,weights)

            print("Predict for the validation data")
            print("Valid_datas:len={}".format(len(valid_datas)))
            valid_start_idx = valid_datas[0][0]
            valid_labels = []
            valid_weights = []
            valid_ds = []
            for d in valid_datas:
                valid_labels.append(d[3])
                valid_weights.append(d[4])
                valid_ds.append(d[2])
                #print("Valid_ds:d.len={},valid_ds.len={}".format(len(d[2]),len(valid_ds)))
            num = len(valid_labels)
            if num > 1:
                valid_weights = np.concatenate(valid_weights,axis=0)
                valid_labels = np.concatenate(valid_labels, axis=0)
                from scipy.sparse import hstack
                #valid_ds = np.concatenate(valid_ds,axis=0)
                valid_ds = hstack(valid_ds,axis=0)
            else:
                valid_labels = valid_labels[0]
                valid_weights = valid_weights[0]
                valid_ds = valid_ds[0]
            y_pred = self.do_thread_execute(predict_batch,self.clf,valid_ds)
            num = len(valid_labels)
            y_pred = np.reshape(y_pred,(num,))
            print("y_pred.shape={}".format(y_pred.shape))
            print("valid_labels.shape={}".format(valid_labels.shape))
            valid_labels = np.reshape(valid_labels,(num,))
            train_preds.append((valid_start_idx,num,y_pred))
            auc_cv[fold] = dist_utils._auc(valid_labels, y_pred)
            logger.info("      {:>3}    {:>8}    {} x {}".format(
                fold+1, np.round(auc_cv[fold],6), valid_ds.shape[0], valid_ds.shape[1]))

            #clean up
            del(valid_datas)
            del(valid_ds)
            del(valid_labels)
            del(valid_weights)
            gc.collect()

        # Save cv result data
        fname = "%s/cv_pred_%s_%s.csv"%(config.OUTPUT_DIR, "fmftrl",Ver)
        print("Save cv predictions:{}".format(fname))
        df = pd.DataFrame({"predicted": all_cv_preds})
        df.to_csv(fname, index=False, columns=["predicted"])


date_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
data_tag = "%s_%s"%(Ver,date_str)

print("===First CV to verify the AUC and ACC, and generate the predicted by CV")
print("====Train model using all the data")
pretrain_files = [
    "../newdata/train_v26_0.csv",
    "../newdata/train_v26_1.csv",
    "../newdata/train_v26_2.csv",
    "../newdata/train_v26_3.csv",
    "../newdata/train_v26_4.csv",
    "../newdata/train_v26_5.csv",
    "../newdata/train_v26_6.csv",
    "../newdata/train_v26_7.1.csv",
]
train_file="../newdata/train_v26_7_9.csv" #["newdata/train_wb_{}.csv".format(i) for i in range(10)]
test_file="../newdata/test_v26_0.csv" #["newdata/train_wb_{}.csv".format(i) for i in range(10)]
model = WBFmFtrlModel(pretrain_files, train_file, test_file)
print("====Train CV Begin")
model.train_cv()
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
