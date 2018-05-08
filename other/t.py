import numpy as np
import pandas as pd
#import ray.dataframe as pd

import sys

"""
click_idx = "data/clicks.idx.sorted"
download_idx = "data/downloads.idx.sorted"

def read_idx(idx_file,num=1000):
	all_idxs = []
	for line in open(idx_file):
		fld = int(line.strip())
		all_idxs.append(fld)
	total = len(all_idxs)
	step = total /num
	idxes = [all_idxs[i* step] for i in range(num)]
	return idxes
		
click_idxs = read_idx(click_idx,num=500)
download_idxs = read_idx(download_idx,num=100)

print(click_idxs)
sys.exit(-1)
"""
#file="data/test_feature_norm_v1_simple.csv"
#data = pd.read_csv(file)
#print("len={}".format(len(data)))

import tensorflow as tf

a = tf.constant([[[1,1,1,1],[2,2,2,2]],[[3,3,3,3],[4,4,4,4]],[[5,5,5,5],[6,6,6,6]]],dtype=tf.float32)
squared_a = tf.reduce_sum(a * a,axis=1)
sum_a = tf.reduce_sum(a,axis=1)
out = tf.reshape(0.5 * tf.reduce_sum(sum_a * sum_a - squared_a,axis=1),[3,1])

b= tf.constant([[2,2,2,2],[1,1,1,1]],dtype=tf.float32)
c = a *b
with tf.Session() as sess:
    av,s1,s2,o,cv,bv = sess.run([a,squared_a,sum_a,out,c,b])
    print(av.shape)
    print(s1.shape)
    print(s2.shape)
    print(o.shape)
    print(o)

    print("a={}".format(av))
    print("b={}".format(bv))
    print("c={}".format(cv))

    print(cv.shape)
