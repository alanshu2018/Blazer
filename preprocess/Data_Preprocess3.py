#coding:

# Preprocess train_data
import pandas as pd
import numpy as np

Ver = "1.0"

#
# Make some big number in the examples small as percentage
#

import sys
fn = sys.argv[1] #"data/train_feature_norm_v1.csv"
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
    res = ",".join(flds[:8])
    for loop in range(loops):
        if loop ==8 or loop >9:
            continue
        start = 8 + 13 * loop
        end = start + 13
        vars = flds[start:end]
        v_t = float(vars[0])
        flds[start + 2 ] = "%.5f"%(float(flds[start+2]))
        res +=",{},{},{},{}".format(vars[0],vars[1],vars[2],','.join(vars[3:9]))
    print(res)

