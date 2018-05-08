# Blazer-- code for TalkingData Adtracking Fraud Detection
Some utilities for kaggel competition "https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection"
In addition the sklearn models, the following models are supported:
* Lightgbm/GBDT
* DeepFM (not working well)
* DeepFMRank (not working well)
* Wordbatch FMFtrl (It depends on the feature selection, may not work well)

## Result
The first place has AUC 0.9834 on Public LB, and 0.98443 on Private LB.
My result is not good, AUC 0.980 on Public LB, and 0.9807 on Private LB.
I learned a lot from the competition, which is more important than the competition itself, isn't it?!

### What I learned
* Train the big model than the memory by enlarging the swap partition
* Important to partition the validation data set properly to make the stacking works
* Wordbatch FM Ftrl model
* Re-implemente my tunner and stacking framework
* Implement Deep models such as DeepFM and DeepFMRank
* Keras implementation for deep learning
* Normalization (batch normal and scale) is important for deep learning models.
Otherwise the models will not converge.

### The most valuable features for fraud detection
* The next_click and prev_click for combination of ip, os, device, app, channel, So click intervals are important
* count features for combination of ip, os, device, app, channel, hour
* unique count features for combination of ip, os, device, app, channel, hour
* accumulated count features for combination of ip, os, device, app, channel, hour
* click mean features for combination of ip, os, device, app, channel, hour
* click variance features for combination of ip, os, device, app, channel, hour
* history download/click ratio, but it will make the trained model overfit. One idea is to try the ratio for
date before the train data for training the model, and use the ratio before test data for predicting. I will try
it some time in the future.

# TalkingData AdTracking Fraud Detection Challenge
## Description
Fraud risk is everywhere, but for companies that advertise online, click fraud can happen at an overwhelming volume, resulting in misleading click data and wasted money. Ad channels can drive up costs by simply clicking on the ad at a large scale. With over 1 billion smart mobile devices in active use every month, China is the largest mobile market in the world and therefore suffers from huge volumes of fradulent traffic.

TalkingData, China’s largest independent big data service platform, covers over 70% of active mobile devices nationwide. They handle 3 billion clicks per day, of which 90% are potentially fraudulent. Their current approach to prevent click fraud for app developers is to measure the journey of a user’s click across their portfolio, and flag IP addresses who produce lots of clicks, but never end up installing apps. With this information, they've built an IP blacklist and device blacklist.

While successful, they want to always be one step ahead of fraudsters and have turned to the Kaggle community for help in further developing their solution. In their 2nd competition with Kaggle, you’re challenged to build an algorithm that predicts whether a user will download an app after clicking a mobile app ad. To support your modeling, they have provided a generous dataset covering approximately 200 million clicks over 4 days!

## Data description
train.csv
test.csv
test_supplement
Data Description
For this competition, your objective is to predict whether a user will download an app after clicking a mobile app advertisement.

### File descriptions
train.csv - the training set
train_sample.csv - 100,000 randomly-selected rows of training data, to inspect data before downloading full set
test.csv - the test set
sampleSubmission.csv - a sample submission file in the correct format
UPDATE: test_supplement.csv - This is a larger test set that was unintentionally released at the start of the competition. It is not necessary to use this data, but it is permitted to do so. The official test data is a subset of this data.
Data fields
Each row of the training data contains a click record, with the following features.

ip: ip address of click.
app: app id for marketing.
device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
os: os version id of user mobile phone
channel: channel id of mobile ad publisher
click_time: timestamp of click (UTC)
attributed_time: if user download the app for after clicking an ad, this is the time of the app download
is_attributed: the target that is to be predicted, indicating the app was downloaded
Note that ip, app, device, os, and channel are encoded.

The test data is similar, with the following differences:

click_id: reference for making predictions
is_attributed: not included

# Utilities
## Tunner

* DataReader --- reader class for reading csv file using pandas
* Tunner --- Main tunner class including feature selection, cross validation, evaluation of model and parameters

* DataPiper ---- reader class for reading csv file into chunks using pandas, Used by SmartTunner or call it directly
* SmartTunner ---- Chunked Tunner. This chunked tunner will not be necessarily give the better result than class Tunner

* Stack*.py and Exp*.py ---- The experiment or stacking experiment for tunning or stacking the models.

* LGBBayesSearchCV.py and BayesSearchCV.py ---- Bayesian Tunner for Lightgbm, but it resembles hyperopt, and why it will not works better than hyperopt?

* Learner.py  --- Basic Learners(models)
* LearnerManager ---- Manager class for learners

## Models
* DeepFMRankNetLearner.py ----- Model DeepFMRankNet
* DeepFMNetLearner.py ----      Model DeepFM
* DeepLRLearner.py    -----     Model LogisticRegression using Keras, but it is similar with DeepFMNetLearner
* FmFtrlLearner.py ------ Model Wordbatch FM Ftrl(https://github.com/anttttti/Wordbatch)

* Isotonic.py --- Isotonic regression using scikit-learn
* SaveBinary.py   ---- save lightgbm dataset into binary output file, but the loading will complain format, why???
* WBFmFtrl*.py ---- Model Wordbatch FM Ftrl(https://github.com/anttttti/Wordbatch)

## Blending
SimpleBlending.py   ---- Simple but effective blending for AUC. But AUC only???

## utils
some utils copy from (https://github.com/ChenglongChen/Kaggle_CrowdFlower)
Most of the Ideas are inspired by this work.

## Others and unsorted (For backup only)
* preprocess ---- Some Data preprocess code
* other --- misc python code

# How to run
## Change settings

from Tunner import *

Change your exp_name, which will be used in logs

```
exp_name = "stack22"

```

## Define your configuration
```
class Stack1Conf(object):
    name = exp_name
    # Define the test data file in a list, using multiple file if stacking
    test = [
        "../newdata/test_v30.csv",
    ]
    # Define the train data file in a list, using multiple file if stacking
    train = [
        "../newdata/train_v30.csv",
    ]
    # Define the test data fields (Here target field click_id is just for reading out click_id) in a list,
    # using multiple file if stacking
    test_predictors = [
        [
            'click_id',
            'ip','app','device','channel','os','hour',
            'ip_d_os_c_app','ip_cu_c','ip_da_cu_h','ip_cu_app',
            'ip_app_cu_os','ip_cu_d','ip_d_os_cu_app','ip_da_co','ip_app_co','ip_d_co','app_chl_co','ip_ch_co',
            'app_d_co','ip_os_co','ip_d_os_co','ip_app_h_co','ip_app_os_h_co','ip_d_h_co','app_chl_h_co','ip_ch_h_co',
            'app_os_h_co','ip_os_h_co','ip_d_os_h_co','ip_da_chl_var_h','ip_chl_var_h','ip_app_os_var_h',
            'ip_app_chl_var_da','ip_app_chl_var_h','app_chl_var_h','ip_app_os_mean_h','app_os_mean_h',
            'ip_channel_prevClick','ip_app_device_os_channel_prevClick','ip_os_device_app_prevClick',
            'ip_app_device_os_channel_nextClick','ip_os_device_nextClick',
            'ip_os_device_app_nextClick',
        ],
    ]
    # Define the train data fields (including target field, e.g click_id) in a list,
    # using multiple file if stacking
    train_predictors = [
        [
            'click_id',
            'ip','app','device','channel','os','hour',
            'ip_d_os_c_app','ip_cu_c','ip_da_cu_h','ip_cu_app',
            'ip_app_cu_os','ip_cu_d','ip_d_os_cu_app','ip_da_co','ip_app_co','ip_d_co','app_chl_co','ip_ch_co',
            'app_d_co','ip_os_co','ip_d_os_co','ip_app_h_co','ip_app_os_h_co','ip_d_h_co','app_chl_h_co','ip_ch_h_co',
            'app_os_h_co','ip_os_h_co','ip_d_os_h_co','ip_da_chl_var_h','ip_chl_var_h','ip_app_os_var_h',
            'ip_app_chl_var_da','ip_app_chl_var_h','app_chl_var_h','ip_app_os_mean_h','app_os_mean_h',
            'ip_channel_prevClick','ip_app_device_os_channel_prevClick','ip_os_device_app_prevClick',
            'ip_app_device_os_channel_nextClick','ip_os_device_nextClick',
            'ip_os_device_app_nextClick',
        ],
    ]
    #Target in data, not used
    target_file = "../newdata/train_v30.csv"
    # Define the target and weight field
    target = 'click_id'
    weight = 'weight'
```


## Run it and view the log
* Tune the model, call function main()
* eval the model, call function eval()
* select features for the model using wrapper method, call function select_feature()

* Logs are kept in directory ../newdata/Log
* Cross validation results are saved in ../newdata/Output
* Submission results are saved in ../newdata/Output/Subm

# Data for running
Run GenerateAllData.py, it will generate some features based on train.csv, test.csv, test_supplement.csv,
and save it in directly ../newdata/