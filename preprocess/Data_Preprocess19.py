#coding:

# Preprocess train_data
import pandas as pd
import numpy as np
import math

Ver = "18.0"
Ver1 = "2.0"


means=[1.866671e+07,5.083459e+04,2.351688e+07,1.809987e+08,4.543603e+06,
             1.215442e+05,2.213557e+06,9.366850e+05,1.703376e+07,1.928687e+05,
             3.039419e+06,3.092410e+02,5.026361e+03,1.170573e+06,2.202677e+07,
	 ]
stds=[ 1.147628e+07, 1.801862e+05, 2.076854e+07, 4.308380e+07, 3.937021e+06,
            1.535289e+05, 2.650926e+06, 6.892881e+05, 1.136252e+07, 2.888697e+05,
            3.759831e+06, 1.343773e+03, 2.044271e+04, 1.112249e+06, 2.028386e+07,
            ]

def float2str1(f):
    if f == 0.0 :
        return "0"
    ret = "%.5f"%(f)
    while ret[-1] == '0':
        ret = ret[:-1]
    if ret[-1] == '.':
        ret = ret[:-1]

    return ret

def float2str(f):
    if f == 0.0 :
        return "0"
    ret = "%.4f"%(f)
    while ret[-1] == '0':
        ret = ret[:-1]
    if ret[-1] == '.':
        ret = ret[:-1]

    return ret

class FeatureStat(object):
    ip_stat_map = {}
    app_stat_map = {}
    os_stat_map = {}
    device_stat_map = {}
    channel_stat_map = {}

    app_channel_hour_map = {}
    app_channel_map = {}
    app_os_hour_map = {}
    app_os_map = {}
    app_device_hour_map = {}
    app_device_map = {}
    #app_os_device_map = {}
    #app_channel_map = {}
    ip_app_hour_map = {}
    ip_app_map = {}
    ip_os_map = {}
    ip_device_map = {}
    ip_channel_map = {}
    os_device_hour_map = {}
    os_device_map = {}

    ip_stat_map1 = {}
    app_stat_map1 = {}
    os_stat_map1 = {}
    device_stat_map1 = {}
    channel_stat_map1 = {}

    app_channel_hour_map1 = {}
    app_channel_map1 = {}
    app_os_hour_map1 = {}
    app_os_map1 = {}
    app_device_hour_map1 = {}
    app_device_map1 = {}
    #app_os_device_map = {}
    #app_channel_map = {}
    ip_app_hour_map1 = {}
    ip_app_map1 = {}
    ip_os_map1 = {}
    ip_device_map1 = {}
    ip_channel_map1 = {}
    os_device_hour_map1 = {}
    os_device_map1 = {}

    def __init__(self):
        self.read_map(self.ip_stat_map ,"newdata/grp_{}.stat".format("ip"))
        self.read_map(self.app_stat_map ,"newdata/grp_{}.stat".format("app"))
        self.read_map(self.os_stat_map ,"newdata/grp_{}.stat".format("os"))
        self.read_map(self.device_stat_map ,"newdata/grp_{}.stat".format("device"))
        self.read_map(self.channel_stat_map ,"newdata/grp_{}.stat".format("channel"))

        self.read_map(self.app_os_hour_map ,"newdata/grp_{}.stat".format("app_os_hour"))
        self.read_map(self.app_os_map ,"newdata/grp_{}.stat".format("app_os"))

        self.read_map(self.app_device_hour_map ,"newdata/grp_{}.stat".format("app_device_hour"))
        self.read_map(self.app_device_map ,"newdata/grp_{}.stat".format("app_device"))

        #self.app_os_device_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_os_device",Ver))
        self.read_map(self.app_channel_hour_map ,"newdata/grp_{}.stat".format("app_channel_hour"))
        self.read_map(self.app_channel_map ,"newdata/grp_{}.stat".format("app_channel"))

        self.read_map(self.ip_app_hour_map ,"newdata/grp_{}.stat".format("ip_app_hour"))
        self.read_map(self.ip_app_map ,"newdata/grp_{}.stat".format("ip_app"))
        #self.ip_os_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_os",Ver))
        #self.ip_device_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_device",Ver))
        #self.ip_channel_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_channel",Ver))
        self.read_map(self.os_device_hour_map ,"newdata/grp_{}.stat".format("os_device_hour"))
        self.read_map(self.os_device_map ,"newdata/grp_{}.stat".format("os_device"))

    def read_map(self, map, filename):
        stat = self.read_grp_stat(filename+"."+Ver)
        stat1 = self.read_grp_stat1(filename+"."+Ver1)
        for k, v in stat.items():
            v1 = stat1.get(k,[0.0])
            map[k] = [v[0],v[1],v1[0],v[3],v[4]]

    # Read stat from grp file and build the map
    def read_grp_stat(self,file):
        map = {}
        for line in open(file):
            flds = line.strip().split(",")
            key = flds[0]
            map[key] = [float(v) for v in flds[1:]]
        return map

    # Read stat from grp file and build the map
    def read_grp_stat1(self,file):
        map = {}
        for line in open(file):
            flds = line.strip().split(",")
            key = flds[0]
            map[key] = [float(flds[3])] #[float(v) for v in flds[1:]]
        return map

    def get_stat(self,map, key):
        stat = map.get(key,None)
        if stat is None:
            stat = map.get("mean")

        return stat

    def get_feature_str(self,stat, stat1):
        #return "{},{},{}".format(stat[0],stat[2],stat[3])
        return "{},{},{}".format(stat[0],stat1[0],stat[3])

    def get_feature_str_by_key(self, map, map1, key):
        stat = self.get_stat(map, key)
        stat1 = self.get_stat(map1, key)
        return "," + self.get_feature_str(stat, stat1)

    def add_stat(self,stat, flds1,flds2,flds3):
        flds1.append(stat[0])
        flds2.append(stat[1])
        flds3.append(stat[2])

    def get_feature_str_by_key1(self, map,key):
        stat = self.get_stat(map, key)
        return stat[0],stat[2],stat[3]

    def get_feature_row(self, row):
        ip = str(int(row["ip"]))
        app = str(int(row["app"]))
        device = str(int(row["device"]))
        channel = str(int(row["channel"]))
        os = str(int(row["os"]))
        day = str(int(row["day"]))
        hour = str(int(row["hour"]))

        res = "{},{},{},{},{},{},{},{}".format(ip,app,device,channel,os,day,hour,int(int(hour)/4))

        flds1=[]
        flds2=[]
        flds3=[]

        stat1= self.get_feature_str_by_key1(self.app_stat_map,app)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.ip_stat_map,ip)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.os_stat_map,os)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.device_stat_map,device)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.channel_stat_map,channel)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.app_os_hour_map,app+"_"+os+"_"+hour)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.app_os_map,app+"_"+os)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.app_device_hour_map,app+"_"+device+"_"+hour)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.app_device_map,app+"_"+device)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1 = self.get_feature_str_by_key1(self.app_channel_hour_map,app + "_" + channel+"_"+hour)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1 = self.get_feature_str_by_key1(self.app_channel_map,app + "_" + channel)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.ip_app_hour_map, ip + "_" + app + "_" + hour)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.ip_app_map, ip + "_" + app)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.os_device_hour_map, os + "_" + device+"_"+hour)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.os_device_map, os + "_" + device)
        self.add_stat(stat1,flds1,flds2,flds3)

        for idx, (f, m, s) in enumerate(zip(flds1,means,stds)):
            flds1[idx] = (f - m)/s
        flds1 = map(lambda f: str(f),flds1) #map(lambda f: math.log10(1+f),flds1))
        flds2 = map(float2str1,flds2)
        flds3 = map(float2str,flds3)
        res += ',' + ','.join(flds1) + "," + ",".join(flds2) + "," + ",".join(flds3)

        return res

feature_stat = FeatureStat()

#train_df = pd.read_hdf("data/train.hdf","data")
import sys
train_df = pd.read_csv(sys.argv[1])
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
len_train = len(train_df)
print("label,ip,app,device,channel,os,day,hour,hour_4," + \
	"c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15," + \
	"r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15," + \
	"h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15")
for i in range(len_train):
    row = train_df.loc[i]
    if row.get("is_attributed",None) is not None:
        label = str(int(row["is_attributed"]))
    else:
        label = row["click_id"]
    feature = feature_stat.get_feature_row(row)
    print("{},{}".format(label,feature))

"""
test_df = pd.read_hdf("data/test.hdf","data")
len_test = len(test_df)
for i in range(len_test):
    row = test_df.iloc[i]
    label = str(int(row["click_id"]))
    feature = feature_stat.get_feature_row(row)
    print("{},{}".format(label,feature))
"""
