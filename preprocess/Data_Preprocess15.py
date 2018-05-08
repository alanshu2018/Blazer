#coding:

# Preprocess train_data
import pandas as pd
import numpy as np
import math

#v13->v15
#label,ip,app,device,channel,os,day,hour,hour_4,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15
def reverse_func(f):
    return str(int(math.pow(10,float(f)) -1))

def process_file(infile,outfile):
    first = True
    print("Input:{},output:{}".format(infile,outfile))
    fout = open(outfile,"w")
    for line in open(infile):
        flds = line.strip().split(",")
        if first == True:
            fout.write(line)
            first = False
            continue
        else:
            flds[9:9+15] = map(reverse_func,flds[9:9+15])
        fout.write(','.join(flds)+"\n")

process_file("newdata/dev1_v13.csv","newdata/dev1_v15.csv")
process_file("newdata/dev2_v13.csv","newdata/dev2_v15.csv")
process_file("newdata/test_v13.csv","newdata/test_v15.csv")
process_file("newdata/train_small_v13.csv","newdata/train_small_v15.csv")

