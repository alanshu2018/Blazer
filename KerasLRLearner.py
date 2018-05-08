#coding:utf-8
#coding:utf-8

import numpy as np
from Learner import BaseLearner
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Input,Embedding, Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras import backend as K
from keras.models import Model
from keras.layers.merge import multiply


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

# -------------------------------------- Keras ---------------------------------------------
# sujianlin's cross entropy
def mycrossentropy(y_true, y_pred, e=0.1,nb_classes=2):
    return (1-e)*K.categorical_crossentropy(y_pred,y_true) + e*K.categorical_crossentropy(y_pred, K.ones_like(y_pred)/nb_classes)

# Kaiming He's focal loss
def focal_loss(y_true, y_pred, e=0.1,nb_classes=2):
    alpha= 0.5 #0.25
    gamma = 2
    return -y_true*alpha*K.pow(1-y_pred,gamma)*K.log(y_pred) - (1-y_true)*(1-alpha)*K.pow(y_pred,gamma)*K.log(1-y_pred)

#callbacks = [EarlyStopping(monitor='val_acc',patience=3,verbose=0,mode='max')]
callbacks = [EarlyStopping(monitor='val_acc',patience=3,verbose=0,mode='max')]
class MyKerasDNNRegressor(object):
    def __init__(self,config):
        self.config = config
        self._build()

    def _build(self):
        X = Input(shape=(3,),name="X")
        #y = Input(shape=(None,),name="y")

        input = Dropout(self.config['input_dropout'])(X)
        for i in range(self.config['hidden_layers']):
            #GLU
            dense = Dense(self.config['hidden_units'])(input)
            gate = Dense(self.config['hidden_units'],activation='sigmoid')(input)
            input = multiply([dense,gate]) #以上三步构成了所谓的GLU激活函数
            #Normal dense
            #input = Dense(self.config['hidden_units'],activation=self.config['hidden_activation'])(input)
            #input = Dropout(self.config['hidden_dropout'])(input)
        predict =  Dense(1,activation='sigmoid')(input)
        model = Model(inputs=X,outputs=predict)
        model.compile(loss = focal_loss, #loss="binary_crossentropy",
                      metrics=['accuracy',auc],
                      #metrics=['accuracy'],
                      optimizer=self.config['optimizer'])
        self.model = model

    def fit(self,X, y, validate=True):
        if validate:
            self.model.fit(X,y,
                       batch_size=self.config['batch_size'],
                       epochs=self.config['nb_epoch'],
                       callbacks = callbacks,
                       verbose=1,validation_split=0.1)
        else:
            self.model.fit(X,y,
                           batch_size=self.config['batch_size'],
                           epochs=self.config['nb_epoch'],
                           verbose=1)
    def predict(self,X):
        return self.model.predict(X)

## regression with Keras' deep neural network
class MyKerasDNNRegressorLearner(BaseLearner):
    name = "reg_my_keras"

    param_space = {
        "input_dropout": hp.quniform("input_dropout", 0, 0.2, 0.05),
        "hidden_layers": hp.quniform("hidden_layers", 1, 3, 1),
        "hidden_units": hp.quniform("hidden_units", 32, 128, 32),
        "hidden_activation": hp.choice("hidden_activation", ["relu", "elu"]),
        "hidden_dropout": hp.quniform("hidden_dropout", 0, 0.5, 0.05),
        "batch_norm": hp.choice("batch_norm", ["before_act", "after_act", "no"]),
        "optimizer": hp.choice("optimizer", ["adam", "adadelta", "rmsprop"]),
        "batch_size": hp.quniform("batch_size", 10000, 20000, 2000),
        "nb_epoch": hp.quniform("nb_epoch", 1, 20, 1),
    }

    param_space1 = {
        "input_dropout": hp.quniform("input_dropout", 0, 0.6, 0.1),
        "hidden_layers": hp.quniform("hidden_layers", 1, 4, 1),
        "hidden_units": hp.quniform("hidden_units", 32, 128, 32),
        "hidden_activation": hp.choice("hidden_activation", ["relu", "elu"]),
        "hidden_dropout": hp.quniform("hidden_dropout", 0, 0.5, 0.05),
        "batch_norm": hp.choice("batch_norm", ["before_act", "after_act", "no"]),
        "optimizer": hp.choice("optimizer", ["adam", "adadelta", "rmsprop"]),
        "batch_size": hp.quniform("batch_size", 10000, 20000, 2000),
        "nb_epoch": hp.quniform("nb_epoch", 1, 20, 1),
    }

    def create_model(self,params):
        self.params = params

        return MyKerasDNNRegressor(params)

