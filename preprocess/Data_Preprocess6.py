#coding:

# Preprocess train_data
import pandas as pd
import numpy as np

Ver = "1.0"

#
# Make some big number in the examples small as percentage
#
#log scale click and download count
import sys
import math

#fn = sys.argv[1] #"data/train_feature_norm_v1.csv"
def preprocess_file(fn,fn1):
    #fn = "data/dev1_feature_norm.simple.csv"
    fout = open(fn1,"w")
    idx = 0
    for line in open(fn):
        idx += 1
        if idx == 1: # copy header
            fout.write(line)
            continue
        #if idx > 100:
        #    break
        flds = line.strip().split(",")
        #if len(flds) < 13:
        #    continue
        #print("line={}".format(line.strip()))
        flds0 = flds[8:11] + flds[17:20] + flds[26:48]
        #print("flds0={}".format(','.join(flds0)))
        flds1 = []
        flds2 = []
        for i in range(len(flds0)):
            if (i+1) % 3==0:
                flds2.append(flds0[i])
            else:
                if flds0[i] == "0":
                    flds1.append("0")
                else:
                    flds1.append("%.4f"%(math.log10(float(flds0[i])+1.0)))

        hour_dist = flds[11:17] + flds[20:26]
        #print("flds1={}".format(','.join(flds1)))
        #print("flds2={}".format(','.join(flds2)))
        flds = flds[0:8] + flds1 + flds2 + hour_dist
        fout.write(",".join(flds)+"\n")
    fout.close()


preprocess_file("data/dev1_feature_norm.simple.csv","data/dev1_feature_norm.simple.v2.csv")
preprocess_file("data/dev2_feature_norm.simple.csv","data/dev2_feature_norm.simple.v2.csv")
preprocess_file("data/test_feature_norm.simple.csv","data/test_feature_norm.simple.v2.csv")
preprocess_file("data/train_feature_norm.simple.small.csv","data/train_feature_norm.simple.small.v2.csv")
preprocess_file("data/train_feature_norm.simple.csv","data/train_feature_norm.simple.v2.csv")
