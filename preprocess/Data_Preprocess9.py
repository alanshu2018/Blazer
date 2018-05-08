#coding:

# Preprocess train_data
import pandas as pd
import numpy as np

import math
import time

Ver = "1.0"

#
# Make some big number in the examples small as percentage
#
click_max = math.log10(306495+1)
down_max = math.log10(1+174330052)


import sys

def float2str(f):
    if f == 0.0 :
        return "0"
    ret = "%.4f"%(f)
    while ret[-1] == '0':
        ret = ret[:-1]
    if ret[-1] == '.':
        ret = ret[:-1]

    return ret

def entropy(probs):
    ret = 0.0
    for prob in probs:
        p = float(prob)
        if p >0:
            ret += -p*math.log(p)
    return float2str(ret)

header = "label,ip,app,device,os,day,hour,hour_4,d1,d2,d3,d4,d5,d6,d7,d8,d9"
header +=",c1,c2,c3,c4,c5,c6,c7,c8,c9,r1,r2,r3,r4,r5,r6,r7,r8,r9,h1,h2,h3,h4,h5,h6,h7,h8,h9"
header +=",m1,m2,m3,m4,m5,m6,m7,m8,m9"

def preprocess_file(fn,fn1):
    #fn = "data/dev1_feature_norm.simple.csv"
    fout = open(fn1,"w")
    fout.write(header+"\n")
    idx = 0
    for line in open(fn):
        idx += 1
        #if idx > 100:
        #    break
        flds = line.strip().split(",")
        if len(flds) < 13:
            continue

        label = flds[0]
        ip = flds[1]
        app = flds[2]
        device = flds[3]
        os = flds[4]
        day = flds[5]
        hour = flds[6]
        hour_4 = flds[7]

        total = len(flds)
        other_total = total -8
        loops = int(other_total /13)

        flds0 = flds[:8]
        flds1 = []
        flds2 = []
        flds3 = []
        flds4 = []
        flds5 = []
        res = ",".join(flds[:8])
        for loop in range(loops):
            if loop ==8 or loop >9:
                continue
            start = 8 + 13 * loop
            end = start + 13
            vars = flds[start:end]
            v_t = float(vars[0])
            flds1.append(float2str(math.log10(float(vars[0])+1.0)/down_max))
            flds2.append(float2str(math.log10(float(vars[1])+1.0)/click_max))
            flds3.append(float2str(float(vars[2])))
            flds4.append(entropy(vars[3:9]))
            flds5.append(entropy(vars[9:13]))
        res +=",{},{},{},{},{}".format(
            ','.join(flds1),
            ','.join(flds2),
            ','.join(flds3),
            ','.join(flds4),
            ','.join(flds5))
        fout.write(res+"\n")

#preprocess_file("data/dev1_feature_norm_v1.csv","data/dev1_feature_norm.simple.v5.csv")
#preprocess_file("data/dev2_feature_norm_v1.csv","data/dev2_feature_norm.simple.v5.csv")
preprocess_file("data/test_feature_norm_v1.csv","data/test_feature_norm.simple.v5.csv")
#preprocess_file("data/train_feature_norm_v1.small.csv","data/train_feature_norm.simple.small.v5.csv")
preprocess_file("data/train_feature_norm_v1.csv","data/train_feature_norm.simple.v5.csv")
