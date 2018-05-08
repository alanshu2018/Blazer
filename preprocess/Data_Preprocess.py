#coding:

# Preprocess train_data
import pandas as pd
import numpy as np

Ver = "2.0"

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

    def __init__(self):
        self.ip_stat_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip",Ver))
        self.app_stat_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app",Ver))
        self.os_stat_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("os",Ver))
        self.device_stat_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("device",Ver))
        self.channel_stat_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("channel",Ver))

        self.app_os_hour_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_os_hour",Ver))
        self.app_os_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_os",Ver))

        self.app_device_hour_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_device_hour",Ver))
        self.app_device_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_device",Ver))

        #self.app_os_device_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_os_device",Ver))
        self.app_channel_hour_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_channel_hour",Ver))
        self.app_channel_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("app_channel",Ver))

        self.ip_app_hour_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_app_hour",Ver))
        self.ip_app_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_app",Ver))
        #self.ip_os_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_os",Ver))
        #self.ip_device_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_device",Ver))
        #self.ip_channel_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("ip_channel",Ver))
        self.os_device_hour_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("os_device_hour",Ver))
        self.os_device_map = self.read_grp_stat("newdata/grp_{}.stat.{}".format("os_device",Ver))

    # Read stat from grp file and build the map
    def read_grp_stat(self,file):
        map = {}
        for line in open(file):
            flds = line.strip().split(",")
            key = flds[0]
            map[key] = [v for v in flds[1:]]
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

    def get_feature_row(self, row):
        ip = str(int(row["ip"]))
        app = str(int(row["app"]))
        device = str(int(row["device"]))
        channel = str(int(row["channel"]))
        os = str(int(row["os"]))
        day = str(int(row["day"]))
        hour = str(int(row["hour"]))

        res = "{},{},{},{},{},{},{},{}".format(ip,app,device,channel,os,day,hour,int(int(hour)/4))

        res += self.get_feature_str_by_key(self.app_stat_map,app)
        res += self.get_feature_str_by_key(self.ip_stat_map,ip)
        res += self.get_feature_str_by_key(self.os_stat_map,os)
        res += self.get_feature_str_by_key(self.device_stat_map,device)
        res += self.get_feature_str_by_key(self.channel_stat_map,channel)
        res += self.get_feature_str_by_key(self.app_os_stat_map,app+"_"+os)
        res += self.get_feature_str_by_key(self.app_device_map,app+"_"+device)
        #res += self.get_feature_str_by_key(self.app_os_device_map,app+"_" + os + "_" + device)
        res += self.get_feature_str_by_key(self.app_channel_map,app + "_" + channel)
        res += self.get_feature_str_by_key(self.ip_app_map, ip + "_" + app)
        #res += self.get_feature_str_by_key(self.ip_os_map,ip + "_" + os)
        #res += self.get_feature_str_by_key(self.ip_device_map,ip + "_" + device)
        #res += self.get_feature_str_by_key(self.ip_channel_map,ip + "_" + channel)
        res += self.get_feature_str_by_key(self.os_device_map, os + "_" + device)

        return res

feature_stat = FeatureStat()

#train_df = pd.read_hdf("data/train.hdf","data")
import sys
train_df = pd.read_csv(sys.argv[1])
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
len_train = len(train_df)
for i in range(len_train):
    row = train_df.loc[i]
    label = str(int(row["is_attributed"]))
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
