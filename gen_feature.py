#coding: utf-8

import os
import sys

import pkgutil
import time_utils
import logging_utils

import config

feature_name = "stack1"

logname = "feature_combiner_%s_%s.log"%(feature_name, time_utils._timestamp())
logger = logging_utils._get_logger(config.LOG_DIR, logname)

data_dict = {
    "train_basic": "newdata/train_v20.csv",
    "train_files": [],
    "test_files": [],
}

fname = os.path.join(config.FEAT_DIR+"/Combine", feature_name+config.FEAT_FILE_SUFFIX)
pkl_utils._save(fname, data_dict)
logger.info("Save to %s" % fname)