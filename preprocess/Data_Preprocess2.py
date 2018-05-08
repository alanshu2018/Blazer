#coding:

# Preprocess train_data
import pandas as pd
import numpy as np

Ver = "1.0"

#
# Make some big number in the examples small as percentage
#
f_train = open("data/t_train.csv","w")
f_test = open("data/t_test.csv","w")

fn = "data/train_feature_norm_v1.csv"
idx = 0
total = 0
total_val = 0
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
    hour = int(flds[6])
    hour_4 = flds[7]

    total += 1
    if total < 120000000:
        f_train.write(line)
    elif hour == 4 or hour == 5 or hour == 9 or hour == 10 or hour == 13 or hour==14:
        total_val += 1
        #if total_val % 100 ==0:
        #    print("total={},val={},line={}".format(total,total_val,line.strip()))
        f_test.write(line)
    else:
        f_train.write(line)

f_test.close()
f_train.close()
print("total={},val={}".format(total,total_val))

