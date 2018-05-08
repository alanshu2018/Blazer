#coding:


basic = [
    'ip',
    'device',
    'os',
    'channel',
    'app',
    #'day',
    'hour',
]
corr_important=[
    'ip_da_cu_h',# 0.184582
    'ip_cu_c',# 0.115164
    'ip',# 0.076869
    'ip_d_os_cu_app',# 0.065968
    'app',# 0.061172
    'ip_app_cu_os',# 0.058503
    'ip_app_chl_var_h',# 0.056425
    'next_click',# 0.053153
    'app_cu_chl',# 0.052197
    'app_os_var_da',# 0.047066
    'ip_da_chl_var_h',# 0.045065
    'app_chl_h_co',# 0.044293
    'app_chl_co',# 0.044271
    'ip_chl_var_h',#_h 0.044056
    'app_d_h_co',# 0.043135
    'ip_cu_app',# 0.03832
    'app_d_co',# 0.037035
    'app_d_var_h',# 0.036244
    'app_os_h_co',# 0.030945
    'ip_app_chl_var_da',# 0.029891
    'app_chl_var_h',# 0.027971
    'ip_app_os_var_h',# 0.025686
    'ip_d_co',# 0.024802
    #'app_os_co',# 0.017825
    #'ip_app_mean_h',# 0.017312
    #'channel',# 0.016108
    #'ip_cu_d',# 0.015942
    #'ip_app_co',# 0.012108
    #'ip_app_chl_mean_h',# 0.011122
]

lgb_important = [
    'channel',
    'ip',
    'next_click',
    'os',
    'app',
    'app_os_var_da',
    'ip_d_os_c_app',
    'app_chl_co',
    'ip_app_co',
    'app_os_h_co',
    'ip_app_cu_os',
    'app_chl_h_co',
    'app_d_co',
    'ip_cu_app',
    'ip_d_co',
    'app_d_var_h',
    'ip_os_h_co',
    'ip_os_co',
    'app_chl_var_h',
    'app_os_co',
    'ip_d_h_co',
    'app_chl_h_co',
    'ip_cu_c',
    'ip_d_os_cu_app',
    'app_os_mean_h',
    #'ip_cu_d',
    #'ip_c_os',
    #'app_mean_var_h',
    #'app_cu_chl',
    #'ip_da_co'
]

important_features = set(basic + corr_important + lgb_important)
print("len={}".format(len(important_features)))
print(important_features)
