#coding: utf-8

import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import *

class DataPiper(object):
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

    def __init__(self,file, logger, chunk_size=10000000, dtypes=None):
        self.file = file
        self.logger = logger
        self.chunk_size = chunk_size

        if dtypes is not None:
            self.dtypes = dtypes

    def get_chunk_data(self):
        rcount = 0

        for df_c in pd.read_csv(self.file, engine='c',
                        chunksize=self.chunk_size,
                        dtype=self.dtypes): #usecols=usecols):

            idx = rcount
            rcount += len(df_c)
            # Return idx, fold_num and data, labels, weight
            yield (idx, df_c) #, df_c[self.predictors],df_c[self.target_name], df_c[self.weight_name])

class DataLoader(object):
    def __init__(self, data_conf, logger, n_fold):
        self.config = data_conf
        self.logger = logger
        self.n_fold = n_fold

        self._load_data()

    def _get_feature_names(self):
        self.logger.info("get_feature_names={}".format(self.all_feature_names))
        return self.all_feature_names

    def _load_data(self):
        self.train_loaders = []
        self.train_predictors = []
        self.all_feature_names =[]
        for idx, (file, predictors) in enumerate(zip(self.config.train,self.config.train_predictors)):
            train_loader = DataPiper(file,self.logger)
            self.train_loaders.append(train_loader)
            self.train_predictors.append(predictors)
            self.all_feature_names.extend(["{}_{}".format(idx,pred) for pred in predictors \
                                           if pred != self.config.target and pred != self.config.weight])
        #Remove target and weight in feature names
        #self.all_feature_names.remove(self.config.weight)
        #self.all_feature_names.remove(self.config.target)

        self.test_loaders = []
        self.test_predictors = []
        for file, predictors in zip(self.config.test,self.config.test_predictors):
            test_loader = DataPiper(file,self.logger)
            self.test_loaders.append(test_loader)
            self.test_predictors.append(predictors)

        self.logger.info("Load data done")

    def get_test_data(self):
        file_size = self.config.test_size
        size_per_nfold = file_size / self.n_fold

        all_loaders = self.test_loaders
        all_predictors = self.test_predictors

        num_loader = len(all_loaders)

        loader0 = all_loaders[0]
        predictors0 = all_predictors[0]
        for (idx0, d0) in loader0.get_chunk_data():
            if num_loader > 1:
                all_ds = [d0]
                for i in range(1,num_loader):
                    all_ds.append(all_loaders[i].get_chunk_data())
                data = []
                labels = None
                weights = None
                for d,predictors in zip(all_ds,all_predictors):
                    #Extract labels
                    predictors_copy = [ p for p in predictors]
                    if labels is not None and self.config.target in predictors:
                        labels = d[self.config.target].values
                        predictors_copy.remove(self.config.target)
                    # Extract weights
                    if weights is not None and self.config.weight in predictors:
                        weights = d[self.config.weight].values
                        predictors_copy.remove(self.config.weight)
                    data.append(d[predictors_copy].values)
                #concat all the datas
                data = np.concatenate(data,axis=1)
                for ds in all_ds:
                    del(ds)
                gc.collect()
            else:
                labels = None
                weights = None
                #Extract labels
                predictors_copy = [ p for p in predictors0]
                if self.config.target in predictors0:
                    labels = d0[self.config.target].values
                    predictors_copy.remove(self.config.target)
                #Extract weight
                if self.config.weight in predictors0:
                    weights = d0[self.config.weight].values
                    predictors_copy.remove(self.config.weight)
                data = d0[predictors_copy].values
                del(d0)
                gc.collect()

            fold_num = idx0 // size_per_nfold
            yield (idx0, fold_num, data, labels, weights)

    def get_train_data(self):
        file_size = self.config.train_size
        size_per_nfold = file_size / self.n_fold

        all_loaders = self.train_loaders
        all_predictors = self.train_predictors

        num_loader = len(all_loaders)

        loader0 = all_loaders[0]
        predictors0 = all_predictors[0]
        for (idx0, d0) in loader0.get_chunk_data():
            if num_loader > 1:
                all_ds = [d0]
                for i in range(1,num_loader):
                    all_ds.append(all_loaders[i].get_chunk_data())
                data = []
                labels = None
                weights = None
                for d,predictors in zip(all_ds,all_predictors):
                    predictors_copy = [ p for p in predictors]
                    #Extract labels
                    if labels is not None and self.config.target in predictors:
                        labels = d[self.config.target].values
                        predictors_copy.remove(self.config.target)
                    # Extract weights
                    if weights is not None and self.config.weight in predictors:
                        weights = d[self.config.weight].values
                        predictors_copy.remove(self.config.weight)
                    data.append(d[predictors_copy].values)
                #concat all the datas
                data = np.concatenate(data,axis=1)
                for ds in all_ds:
                    del(ds)
                gc.collect()
            else:
                labels = None
                weights = None
                #Extract labels
                predictors_copy = [ p for p in predictors0]
                if self.config.target in predictors0:
                    labels = d0[self.config.target].values
                    predictors_copy.remove(self.config.target)
                #Extract weight
                if self.config.weight in predictors0:
                    weights = d0[self.config.weight].values
                    predictors_copy.remove(self.config.weight)
                data = d0[predictors_copy].values
                del(d0)
                gc.collect()

            fold_num = idx0 // size_per_nfold
            yield (idx0, fold_num, data, labels, weights)

    def get_sampled_train_data(self, predictors, sample_ratio=0.01):
        """
        Sample some train data
        :param predictors:
        :param file_size:
        :param sample_ratio:
        :return:
        """
        file_size = self.config.train_size
        cols = []
        for idx, p in enumerate(self.all_feature_names):
            if p in predictors:
                cols.append(idx)
        for (idx, fold_num, data, labels, weights) in self.get_train_data(file_size):
            len_train = len(data)
            indexes = np.random.permutation(range(len_train))
            sampled_len = int(len_train * sample_ratio)
            indexes = indexes[:sampled_len]
            Xslice = data[indexes]
            yslice = labels[indexes]
            Wslice = weights[indexes]

        return Xslice[:,cols], yslice, Wslice



