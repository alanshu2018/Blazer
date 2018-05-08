#coding:

# Preprocess train_data
import pandas as pd
import numpy as np

Ver = "1.0"

#
# Make some big number in the examples small as percentage
#

import sys
#fn = sys.argv[1] #"data/train_feature_norm_v1.csv"
fn = "data/dev1_feature_norm_v1.csv"
idx = 0
for line in open(fn):
    idx += 1
    if idx == 1: # copy header
        print(line.strip())
        continue
    #if idx > 100:
    #    break
    flds = line.strip().split(",")
    #if len(flds) < 13:
    #    continue
    flds0 = flds[8:11] + flds[17:20] + flds[26:47]
    flds1 = []
    flds2 = []
    for i in range(len(flds1)):
        if i>0 and i % 2==0:
            flds1.append(flds1[i])
        else:
            flds2.append(flds1[i])

    hour_dist = flds[11:17] + flds[20:26]
    flds = flds[0:8] + flds1 + flds2 + hour_dist
    print(",".join(flds))

