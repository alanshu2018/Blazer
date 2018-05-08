# good day, my friends
# in this kernel we try to continue development of our DL models
# thanks for people who share his works. i hope together we can create smth interest

# https://www.kaggle.com/baomengjiao/embedding-with-neural-network
# https://www.kaggle.com/gpradk/keras-starter-nn-with-embeddings
# https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-auc-0-9787
# https://www.kaggle.com/rteja1113/lightgbm-with-count-features
# https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl
# https://www.kaggle.com/isaienkov/rnn-with-keras-ridge-sgdr-0-43553
# https://www.kaggle.com/valkling/mercari-rnn-2ridge-models-with-notes-0-42755/versions#base=2202774&new=2519287


#======================================================================================
# we continue our work started in previos kernel "Deep learning support.."
# + we will try to find a ways which can help us increase specialisation of neural network on our task.
# If you need a details about what we try to create follow the comments

print ('Good luck')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['OMP_NUM_THREADS'] = '4'
import gc

path = 'data/'
dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}
print('load train....')
# we save only day 9
#train_df = pd.read_csv(path+"train.csv", dtype=dtypes, skiprows = range(1, 131886954), usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
train_df = pd.read_hdf(path+"train.hdf","data",start=131886954)

"""
train_df = train_df[
    (train_df.hour==4)|(train_df.hour==5)| \
    (train_df.hour==9)|(train_df.hour==10) \
    |(train_df.hour==13)|(train_df.hour==14)
]
"""

print('load test....')
#test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
test_df = pd.read_hdf(path+"test.hdf","data")
len_train = len(train_df)
#train_df=train_df.append(test_df)
#del test_df; gc.collect()

y_train = train_df['is_attributed'].values
#train_df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)

print ('neural network....')
from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping

##########################################################################
#
# AUC
#
########################################################################
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_auc_score

# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

##########################################################################
max_app = np.max([train_df['app'].max(), test_df['app'].max()])+1
max_ch = np.max([train_df['channel'].max(), test_df['channel'].max()])+1
max_dev = np.max([train_df['device'].max(), test_df['device'].max()])+1
max_os = np.max([train_df['os'].max(), test_df['os'].max()])+1
max_h = np.max([train_df['hour'].max(), test_df['hour'].max()])+1
max_d = np.max([train_df['day'].max(), test_df['day'].max()])+1
max_wd = np.max([train_df['wday'].max(), test_df['wday'].max()])+1
max_qty = np.max([train_df['qty_hour'].max(), test_df['qty_hour'].max()])+1
max_c1 = np.max([train_df['ip_app_count'].max(), test_df['ip_app_count'].max()])+1
max_c2 = np.max([train_df['ip_app_os_count'].max(), test_df['ip_app_os_count'].max()])+1
def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'd': np.array(dataset.day),
        'wd': np.array(dataset.wday),
        'qty': np.array(dataset.qty_hour),
        'c1': np.array(dataset.ip_app_count),
        'c2': np.array(dataset.ip_app_os_count)
    }
    return X
train_df = get_keras_data(train_df)

emb_n = 50
dense_n = 1000
drop_prob = 0.2
batch_size = 20000
epochs = 10
in_app = Input(shape=[1], name = 'app')
emb_app = Embedding(max_app, emb_n)(in_app)
in_ch = Input(shape=[1], name = 'ch')
emb_ch = Embedding(max_ch, emb_n)(in_ch)
in_dev = Input(shape=[1], name = 'dev')
emb_dev = Embedding(max_dev, emb_n)(in_dev)
in_os = Input(shape=[1], name = 'os')
emb_os = Embedding(max_os, emb_n)(in_os)
in_h = Input(shape=[1], name = 'h')
emb_h = Embedding(max_h, emb_n)(in_h)
in_d = Input(shape=[1], name = 'd')
emb_d = Embedding(max_d, emb_n)(in_d)
in_wd = Input(shape=[1], name = 'wd')
emb_wd = Embedding(max_wd, emb_n)(in_wd)
in_qty = Input(shape=[1], name = 'qty')
emb_qty = Embedding(max_qty, emb_n)(in_qty)
in_c1 = Input(shape=[1], name = 'c1')
emb_c1 = Embedding(max_c1, emb_n)(in_c1)
in_c2 = Input(shape=[1], name = 'c2')
emb_c2 = Embedding(max_c2, emb_n)(in_c2)
fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h),
                  (emb_d), (emb_wd), (emb_qty), (emb_c1), (emb_c2)])
fe = SpatialDropout1D(drop_prob)(fe)
gl = GlobalAveragePooling1D()(fe)
avr = GlobalMaxPooling1D()(fe)
fe2 = concatenate([(gl), (avr)])
fe1 = Flatten()(fe)
x = concatenate([(fe1), (fe2)])
#x = BatchNormalization()(x)
x = Dropout(drop_prob)(Dense(dense_n,activation='relu')(x))
#x = BatchNormalization()(x)
x = Dropout(drop_prob)(Dense(dense_n,activation='relu')(x))
outp = Dense(1,activation='sigmoid')(x)
model = Model(inputs=[in_app,in_ch,in_dev,in_os,in_h,in_d,in_wd,in_qty,in_c1,in_c2], outputs=outp)

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(train_df) / batch_size) * epochs
lr_init, lr_fin = 0.0005, 0.00001
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=lr_init, decay=lr_decay)
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy',auc])
#model.compile(loss='hinge',optimizer=optimizer_adam,metrics=['accuracy',auc])

model.summary()

callbacks = [EarlyStopping(monitor='val_auc',patience=5,verbose=0,mode='max')]
class_weight = {0:.01,1:.99} # magic
model.fit(train_df, y_train, batch_size=batch_size, 
        epochs=epochs, class_weight=class_weight,
        shuffle=True, 
        callbacks = callbacks,
        verbose=1,validation_split=0.1)
del train_df, y_train; gc.collect()
model.save_weights('imbalanced_data.h5')

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
#test_df.drop(['click_id', 'click_time','ip','is_attributed'],1,inplace=True)
test_df = get_keras_data(test_df)

print("predicting....")
sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
del test_df; gc.collect()
print("writing....")
sub.to_csv('imbalanced_data.csv',index=False)
