#coding: utf-8

import pandas as pd

import sys

filename=sys.argv[1]
#data = pd.read_csv(filename,skiprows=next_read_pos,nrows=40000000)
data = pd.read_csv(filename)
print(data.describe())
