#coding:

# Preprocess train_data
import pandas as pd
import numpy as np
import math

Ver = "18.0"
class FeatureStat(object):
    def __init__(self):
        self.ip_stat_map = {}
        self.app_stat_map = {}
        self.os_stat_map = {}
        self.device_stat_map = {}
        self.channel_stat_map = {}

        self.app_channel_hour_map = {}
        self.app_channel_map = {}
        self.app_os_hour_map = {}
        self.app_os_map = {}
        self.app_device_hour_map = {}
        self.app_device_map = {}
        self.ip_app_hour_map = {}
        self.ip_app_map = {}
        self.ip_os_map = {}
        self.ip_device_map = {}
        self.ip_channel_map = {}
        self.os_device_hour_map = {}
        self.os_device_map = {}

        self.read_stat()

    def get_stat(self,map, key):
        stat = map.get(key,None)
        if stat is None:
            stat = ""

        return stat

    # Read stat from grp file and build the map
    def read_grp_stat(self,file):
        map = {}
        for line in open(file):
            flds = line.strip().split(",")
            key = flds[0]
            map[key] = float(flds[3]) #[float(v) for v in flds[1:]]
        return map

    def read_map(self, map, filename):
        stat = self.read_grp_stat(filename+"."+Ver)
        for k, v in stat.items():
            map[k] = v

    def read_stat(self):
        self.read_map(self.ip_stat_map ,"newdata/grp_{}.stat".format("ip"))
        self.read_map(self.app_stat_map ,"newdata/grp_{}.stat".format("app"))
        self.read_map(self.os_stat_map ,"newdata/grp_{}.stat".format("os"))
        self.read_map(self.device_stat_map ,"newdata/grp_{}.stat".format("device"))
        self.read_map(self.channel_stat_map ,"newdata/grp_{}.stat".format("channel"))

        self.read_map(self.app_os_hour_map ,"newdata/grp_{}.stat".format("app_os_hour"))
        self.read_map(self.app_os_map ,"newdata/grp_{}.stat".format("app_os"))

        self.read_map(self.app_device_hour_map ,"newdata/grp_{}.stat".format("app_device_hour"))
        self.read_map(self.app_device_map ,"newdata/grp_{}.stat".format("app_device"))

        self.read_map(self.app_channel_hour_map ,"newdata/grp_{}.stat".format("app_channel_hour"))
        self.read_map(self.app_channel_map ,"newdata/grp_{}.stat".format("app_channel"))

        self.read_map(self.ip_app_hour_map ,"newdata/grp_{}.stat".format("ip_app_hour"))
        self.read_map(self.ip_app_map ,"newdata/grp_{}.stat".format("ip_app"))
        self.read_map(self.os_device_hour_map ,"newdata/grp_{}.stat".format("os_device_hour"))
        self.read_map(self.os_device_map ,"newdata/grp_{}.stat".format("os_device"))

fs = FeatureStat()

def get_feature_str_by_key(map,key):
    stat = fs.get_stat(map, key)
    return stat
"""
app,channel,click_id1,click_time,device,ip,click_id,os,hour,ip_d_os_c_app,ip_c_os,ip_cu_c,ip_da_cu_h,ip_cu_app,ip_app_cu_os,ip_cu_d,app_cu_chl,ip_d_os_cu_app,ip_da_co,ip_app_co,ip_app_os_co,ip_d_co,app_chl_co,ip_ch_co,ip_app_chl_co,app_d_co,app_os_co,ip_os_co,ip_d_os_co,ip_app_h_co,ip_app_os_h_co,ip_d_h_co,app_chl_h_co,ip_ch_h_co,ip_app_chl_h_co,app_d_h_co,app_os_h_co,ip_os_h_co,ip_d_os_h_co,ip_da_chl_var_h,ip_chl_var_h,ip_app_os_var_h,ip_app_chl_var_da,ip_app_chl_var_h,app_os_var_da,app_d_var_h,app_chl_var_h,ip_app_chl_mean_h,ip_chl_mean_h,ip_app_os_mean_h,ip_app_mean_h,app_os_mean_h,app_mean_var_h,app_chl_mean_h,ip_channel_prevClick,ip_os_prevClick,ip_app_device_os_channel_prevClick,ip_os_device_prevClick,ip_os_device_app_prevClick,ip_app_device_os_channel_nextClick,ip_os_device_nextClick,ip_os_device_app_nextClick,device_nextClick,device_channel_nextClick,app_device_channel_nextClick,device_hour_nextClick
"""
def process_file(infile,outfile):
    fd = open(outfile,"w")
    with open(infile) as f:
        first = True
        for line in f:
            app,channel,click_id1,click_time,device,ip,click_id,\
            os,hour,ip_d_os_c_app,ip_c_os,ip_cu_c,ip_da_cu_h,ip_cu_app, \
            ip_app_cu_os,ip_cu_d,app_cu_chl,ip_d_os_cu_app,ip_da_co,ip_app_co,\
            ip_app_os_co,ip_d_co,app_chl_co,ip_ch_co,ip_app_chl_co,app_d_co,app_os_co,\
            ip_os_co,ip_d_os_co,ip_app_h_co,ip_app_os_h_co,ip_d_h_co,\
            app_chl_h_co,ip_ch_h_co,ip_app_chl_h_co,app_d_h_co,app_os_h_co,\
            ip_os_h_co,ip_d_os_h_co,ip_da_chl_var_h,ip_chl_var_h,ip_app_os_var_h,\
            ip_app_chl_var_da,ip_app_chl_var_h,app_os_var_da,app_d_var_h,app_chl_var_h,\
            ip_app_chl_mean_h,ip_chl_mean_h,ip_app_os_mean_h,ip_app_mean_h,app_os_mean_h,\
            app_mean_var_h,app_chl_mean_h,ip_channel_prevClick,ip_os_prevClick,\
            ip_app_device_os_channel_prevClick,ip_os_device_prevClick,ip_os_device_app_prevClick,\
            ip_app_device_os_channel_nextClick,ip_os_device_nextClick,ip_os_device_app_nextClick,\
            device_nextClick,device_channel_nextClick,app_device_channel_nextClick,device_hour_nextClick = line.strip().split(",")
            if len(click_id1) > 0:
                # For test , we use click_id1, and test, use clickid
                click_id = str(int(click_id1))
            ret = [
                click_id,
                app_cu_chl, ip, app, ip_cu_c, ip_da_chl_var_h, ip_app_os_var_h, device, app_d_h_co, #8
                app_chl_var_h, ip_d_os_c_app, app_chl_h_co, app_d_co, ip_d_co, app_os_var_da, #7
                ip_da_cu_h, channel, app_os_h_co, app_chl_co, ip_app_chl_var_h, ip_os_co, ip_app_co, #7
                app_os_co, ip_d_os_cu_app, app_d_var_h, ip_chl_var_h, hour, ip_app_cu_os, ip_d_h_co, #7
                ip_app_chl_var_da, app_os_mean_h, ip_cu_app, ip_os_h_co, os, #6
                ip_channel_prevClick,ip_os_prevClick,ip_app_device_os_channel_nextClick,
                ip_os_device_nextClick,ip_os_device_app_nextClick,
            ]

            if first:
                first = False
                ret.extend(["r1","r2","r3","r4","r5","r6","r7","r8","r9","r10","r11","r12","r13","r14","r15"])
            else:
                ret.append(get_feature_str_by_key(fs.app_stat_map,app))
                ret.append(get_feature_str_by_key(fs.ip_stat_map,ip))
                ret.append(get_feature_str_by_key(fs.os_stat_map,os))
                ret.append(get_feature_str_by_key(fs.device_stat_map,device))
                ret.append(get_feature_str_by_key(fs.channel_stat_map,channel))
                ret.append(get_feature_str_by_key(fs.app_os_hour_map,app+"_"+os+"_"+hour))
                ret.append(get_feature_str_by_key(fs.app_os_map,app+"_"+os))
                ret.append(get_feature_str_by_key(fs.app_device_hour_map,app+"_"+device+"_"+hour))
                ret.append(get_feature_str_by_key(fs.app_device_map,app+"_"+device))
                ret.append(get_feature_str_by_key(fs.app_channel_hour_map,app + "_" + channel+"_"+hour))
                ret.append(get_feature_str_by_key(fs.app_channel_map,app + "_" + channel))
                ret.append(get_feature_str_by_key(fs.ip_app_hour_map, ip + "_" + app + "_" + hour))
                ret.append(get_feature_str_by_key(fs.ip_app_map, ip + "_" + app))
                ret.append(get_feature_str_by_key(fs.os_device_hour_map, os + "_" + device+"_"+hour))
                ret.append(get_feature_str_by_key(fs.os_device_map, os + "_" + device))
            res = ",".join(ret)
            fd.write(res+"\n")


Ver = "18.0"

process_file("newdata/train_v28.csv","newdata/train_v29.csv")
process_file("newdata/test_v28.csv","newdata/test_v29.csv")
