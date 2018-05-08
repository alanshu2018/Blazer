#coding:

# Preprocess train_data
import pandas as pd
import numpy as np
import math

Ver = "2.0"

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

    app_os_stat_map = {}
    app_device_map = {}
    app_os_device_map = {}
    app_channel_map = {}
    ip_app_map = {}
    ip_os_map = {}
    ip_device_map = {}
    ip_channel_map = {}
    os_device_map = {}

    def __init__(self):
        #self.ip_stat_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip",Ver))
        #self.app_stat_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app",Ver))
        #self.os_stat_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("os",Ver))
        #self.device_stat_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("device",Ver))
        #self.channel_stat_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("channel",Ver))

        self.app_os_stat_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_os",Ver))
        self.app_device_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_device",Ver))
        #self.app_os_device_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_os_device",Ver))
        self.app_channel_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_channel",Ver))
        self.ip_app_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_app",Ver))
        #self.ip_os_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_os",Ver))
        #self.ip_device_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_device",Ver))
        #self.ip_channel_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_channel",Ver))
        self.os_device_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("os_device",Ver))

    # Read stat from grp file and build the map
    def read_grp_stat(self,file):
        map = {}
        for line in open(file):
            flds = line.strip().split(",")
            key = flds[0]
            map[key] = [float(v) for v in flds[1:]]
        return map

    def get_stat(self,map, key):
        stat = map.get(key,None)
        if stat is None:
            stat = map.get("mean")

        return stat

    def get_feature_str(self,stat):
        return "{},{},{}".format(stat[0],stat[2],stat[3])

    def get_feature_str_by_key(self, map, key):
        stat = self.get_stat(map, key)
        return "," + self.get_feature_str(stat)

    def add_stat(self,stat, flds1,flds2,flds3):
        flds1.append(stat[0])
        flds2.append(stat[1])
        flds3.append(stat[2])

    def get_feature_str_by_key1(self, map, key):
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
        #res += self.get_feature_str_by_key(self.app_stat_map,app)
        #res += self.get_feature_str_by_key(self.ip_stat_map,ip)
        #res += self.get_feature_str_by_key(self.os_stat_map,os)
        #res += self.get_feature_str_by_key(self.device_stat_map,device)
        #res += self.get_feature_str_by_key(self.channel_stat_map,channel)
        stat1= self.get_feature_str_by_key1(self.app_os_stat_map,app+"_"+os)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.app_device_map,app+"_"+device)
        self.add_stat(stat1,flds1,flds2,flds3)
        #res += self.get_feature_str_by_key(self.app_os_device_map,app+"_" + os + "_" + device)
        stat1 = self.get_feature_str_by_key1(self.app_channel_map,app + "_" + channel)
        self.add_stat(stat1,flds1,flds2,flds3)

        stat1= self.get_feature_str_by_key1(self.ip_app_map, ip + "_" + app)
        self.add_stat(stat1,flds1,flds2,flds3)
        #res += self.get_feature_str_by_key(self.ip_os_map,ip + "_" + os)
        #res += self.get_feature_str_by_key(self.ip_device_map,ip + "_" + device)
        #res += self.get_feature_str_by_key(self.ip_channel_map,ip + "_" + channel)
        stat1= self.get_feature_str_by_key1(self.os_device_map, os + "_" + device)
        self.add_stat(stat1,flds1,flds2,flds3)
        flds1 = map(float2str,map(lambda f: math.log10(1+f),flds1))
        flds2 = map(float2str,flds2)
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
print("label,ip,app,device,channel,os,day,hour,hour_4,c1,c2,c3,c4,c5,r1,r2,r3,r4,r5,h1,h2,h3,h4,h5")
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
