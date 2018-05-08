#coding: utf-8

import pandas as pd
import numpy as np
import gc

from sklearn.model_selection import *

class DataReader(object):
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

    def __init__(self, data_conf, logger, n_fold):
        self.config = data_conf
        self.logger = logger
        self.n_fold = n_fold

        self._load_data()

    def _get_feature_names(self):
        self.logger.info("get_feature_names={}".format(self.feature_names))
        return self.feature_names

    def _load_data(self):
        # Read target
        if hasattr(self.config,'skiprows') == False:
            self.config.skiprows = None

        """
        self.logger.info("Read target:{}".format(self.config.target_file))
        target_df = pd.read_csv(
            self.config.target_file,
            skiprows=self.config.skiprows,
            usecols=[self.config.target],
            dtype=self.dtypes)
        self.y_tr = target_df[self.config.target].values
        del(target_df)
        gc.collect()
        """

        # Read Train data
        self.weight_tr = None
        self.y_tr = None
        train_datas = []
        self.feature_names = []
        for idx, (file, predictors) in enumerate(zip(self.config.train,self.config.train_predictors)):
            self.logger.info("Read train :{}".format(file))
            df = pd.read_csv(file,
                             #nrows = 1000000, #FORTEST
                             skiprows=self.config.skiprows,
                             usecols=predictors,dtype=self.dtypes)
            if self.config.weight in predictors and self.weight_tr is None:
                self.weight_tr = df[self.config.weight].values
                predictors.remove(self.config.weight)
            if self.config.target in predictors and self.y_tr is None:
                self.y_tr = df[self.config.target].values
                predictors.remove(self.config.target)

            self.logger.info("Read train :len={}".format(len(df)))
            d = df[predictors].values
            self.feature_names.extend(["{}_{}".format(idx,p) for p in predictors])
            del(df)
            gc.collect()
            print("file={},shape={}".format(file,d.shape))
            if d.shape[1] > 0:
                train_datas.append(d)
        self.logger.info("Feature_names={}".format(self.feature_names))
        if len(train_datas) > 1:
            self.X_tr = np.concatenate(train_datas,axis=1)
        else:
            self.X_tr = train_datas[0]

        # Read test data
        self.weight_te = None
        self.y_te = None
        test_datas = []
        for file, predictors in zip(self.config.test,self.config.test_predictors):
            self.logger.info("Read test :{}".format(file))
            print("Read test :{}".format(file))
            df = pd.read_csv(file,
                             #skiprows=self.config.skiprows, # Do not skip for test
                             usecols=predictors,
							 dtype=self.dtypes)

            if self.config.weight in predictors and self.weight_te is None:
                self.weight_te = df[self.config.weight].values
                predictors.remove(self.config.weight)
            if self.config.target in predictors and self.y_te is None:
                self.y_te = df[self.config.target].values
                predictors.remove(self.config.target)
            d = df[predictors].values
            #click_id = df['click_id'].values
            del(df)
            gc.collect()
            test_datas.append(d)
        if len(test_datas) > 1:
            self.X_te = np.concatenate(test_datas,axis=1)
        else:
            self.X_te = test_datas[0]
        #self.y_te = click_id

        self.len_train = len(self.X_tr)
        self.len_test = len(self.X_te)
        self.logger.info("Load data done")

    def get_sub_data(self, predictors, sample_ratio=0.01):
        cols = []
        for idx, p in enumerate(self.feature_names):
            if p in predictors:
                cols.append(idx)
        indexes = np.random.permutation(range(self.len_train))
        sampled_len = int(self.len_train * sample_ratio)
        indexes = indexes[:sampled_len]
        Xslice = self.X_tr[indexes]
        yslice = self.y_tr[indexes]
        if self.weight_tr is not None:
            Wslice = self.weight_tr[indexes]
        else:
            Wslice = None

        return Xslice[:,cols], yslice, Wslice

    ## for CV
    def _get_train_valid_data(self):
        kf = KFold(n_splits=self.n_fold)
        for train,test in kf.split(self.X_tr):
            W_train = None
            W_test = None
            if self.weight_tr is not None:
                W_train = self.weight_tr[train]
                W_test = self.weight_tr[test]
            yield (self.X_tr[train],self.y_tr[train],self.X_tr[test],self.y_tr[test],train,test, W_train, W_test)

    def _get_test_data(self):
        return self.X_te, self.y_te, self.weight_te

    ## for refit
    def _get_train_test_data(self):
        # feature
        return self.X_tr, self.y_tr, self.X_te, self.y_te, self.weight_tr, self.weight_te

