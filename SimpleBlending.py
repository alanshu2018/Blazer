# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

out_dir = "../newdata/Output/"
sub_dir = "../newdata/Output/Subm/"

models = [
    {
        'name':'lgb_v20',
        'file': sub_dir + 'sub_v20_20180419002702_lgb_lr_0.2_num_leaves_7_max_depth_3.csv',
        'score':97.69,
    },{
        'name':'wb_fm_ftrl_v16_1',
        'file': sub_dir + 'sub_wordbatch_fm_ftrl_16_20180417231311.csv',
        'score':97.69,
    },
    {
        'name':'wb_fm_ftrl_v16_2',
        'file': sub_dir + 'sub_wordbatch_fm_ftrl_wordbatch_fm_ftrl_16.csv',
        'score':97.69,
    },{
        'name':'lgb_stack10',
        'file': sub_dir + 'sub_pred.Feat@stack10_Learner@reg_lgb_native.[Mean0.022502]_[Std0.003067].csv',
        'score':97.00,
    },{
        'name':'lgb_stack13',
        'file': sub_dir + 'sub_pred.Feat@stack13_Learner@reg_lgb_native.[Mean0.017455]_[Std0.002860].csv',
        'score':97.93,
    },{
        'name':'lgb_stack1',
        'file': sub_dir + 'sub_pred.Feat@stack1_Learner@reg_lgb_native.[Mean0.018308]_[Std0.003611].csv',
        'score':97.71,
    },{
        'name':'keras_stack1',
        'file': sub_dir + 'sub_pred.Feat@stack1_Learner@reg_my_keras.[Mean0.017438]_[Std0.002688].csv',
        'score':97.79,
    },{
        'name':'lgb_stack4',
        'file': sub_dir + 'sub_pred.Feat@stack4_Learner@reg_lgb_native.[Mean0.000000]_[Std0.000000].csv',
        'score':97.78,
    },{
        'name':'lgb_stack8',
        'file': sub_dir + 'sub_pred.Feat@stack8_Learner@reg_lgb_native.[Mean0.018597]_[Std0.003297].csv',
        'score':97.82,
    },{
        'name':'fmftrl_22_1',
        'file': sub_dir + 'sub_pred_fmftrl_22.csv',
        'score':97.07,
    },{
        'name':'fmftrl_26_1',
        'file': sub_dir + 'sub_pred_fmftrl_26_20180503172030.csv',
        'score':94.96,
    },{
      'name':'sub_it_v28_40m',
      'file': sub_dir + 'sub_it_v28_40m.csv',
      'score':97.83,
    },{
      'name':'sub_v22.csv',
      'file': "sub_pred.v22_20180507200130.csv",
      'score':97.76,
    },{
      'name':'sub_v30.csv',
      'file': 'sub_pred.v30_20180508032203.csv',
      'score':97.76,
    },
]

sub_am = pd.DataFrame()

# Any results you write to the current directory are saved as output.
print("Reading the data...\n")
num_model = len(models)

isa_lg = 0
isa_hm = 0
isa_am=0
#dfs=[]
click_ids = None
for model in models:
    print("Read model:name={},file={}".format(model["name"],model["file"]))
    df = pd.read_csv(model["file"])
    print("data.shape={}".format(df.shape))
    isa_lg += np.log(df.is_attributed)
    isa_hm += 1/(df.is_attributed)
    isa_am +=df.is_attributed

    if click_ids is None:
        click_ids = 1
        sub_am['click_id'] = df['click_id'].values
    del(df)
    gc.collect()

print("Blending...\n")
isa_lg = np.exp(isa_lg/num_model)
isa_hm = 1/isa_hm
isa_am=isa_am/num_model

print("Isa log\n")
print(isa_lg[:5])
print()
print("Isa harmo\n")
print(isa_hm[:5])


import time
Ver = "v66"
date_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
data_tag = "%s_%s"%(Ver,date_str)

sub_am['is_attributed'] = isa_am
print(sub_am.head())
print("Writing avg...")
sub_am.to_csv('submission_avg_{}.csv'.format(data_tag), index=False, float_format='%.9f')

isa_fin=(isa_am+isa_hm+isa_lg)/num_model

sub_fin = pd.DataFrame()
sub_fin['click_id'] = sub_am['click_id']
sub_fin['is_attributed'] = isa_fin
print("Writing fin...")
sub_fin.to_csv('submission_fin_{}.csv'.format(data_tag), index=False, float_format='%.9f')
