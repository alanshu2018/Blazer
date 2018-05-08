#coding:utf-8

import numpy as np
from Learner import BaseLearner
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Input,Embedding, Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D, Lambda, Flatten
from keras.datasets import imdb
from keras import backend as K
from keras.models import Model
from keras.layers.merge import multiply, add
from keras.optimizers import Adam,SGD,RMSprop


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

focal_loss = lambda y_true,y_pred: y_true*K.relu(0.8-y_pred)**2 + 0.01*(1-y_true)*K.relu(y_pred-0.1)**2
"""
# Kaiming He's focal loss
def focal_loss(y_true, y_pred, e=0.1,nb_classes=2):
    alpha= 0.5 #0.25
    gamma = 2
    return -y_true*alpha*K.pow(1-y_pred,gamma)*K.log(y_pred) - (1-y_true)*(1-alpha)*K.pow(y_pred,gamma)*K.log(1-y_pred)
"""
#callbacks = [EarlyStopping(monitor='val_acc',patience=3,verbose=0,mode='max')]
callbacks = [EarlyStopping(monitor='val_acc',patience=50,verbose=0,mode='max')]

class DeepFMModel(object):
    def __init__(self,config):
        self.config = config
        self._build()

    def build_new_input(self, input):
        """
        Turn input into matrix in order in build_embedding_input
        :param input:
        :return:
        """
        int_output = K.ones_like(input[:,:self.int_dim],dtype='float32')
        float_output = input[:,self.int_dim:]
        return K.concatenate([int_output, float_output],axis=1)

    def build_embedding_input(self, input):
        """
        Build input for embedding module
        The int_dim is accumulated
        :param input:
        :return:
        """
        outputs = []
        acc_num = 0
        for i in range(self.int_dim):
            output = K.cast(input[:,i] + acc_num,'int32')
            outputs.append(K.expand_dims(output,1))
            acc_num += self.feature_sizes[i]
        for i in range(self.float_dim):
            output = K.ones_like(input[:,i],dtype='int32')*acc_num
            outputs.append(K.expand_dims(output,1))
            acc_num += 1
        outputs = K.concatenate(outputs,axis=1)
        return outputs

    #def FM_part(self, x_embedding, V_embedding, new_input):
    def FM_part(self, X ):
         x_embedding, V_embedding, new_input = X
         # x_embeding: [None, all_dim, 1]
         # new_input: [None, all_dim]
         y_FM = K.sum(x_embedding * K.expand_dims(new_input,2), axis=[1,2])
         # new_input [ None, all_dim]
         # V_embedding [None, all_dim, embed_size]
         # xv [None, all_dim, embed_size]
         xv = V_embedding * K.expand_dims(new_input,2)
         y_FM += 0.5 * K.mean(
             K.square(K.sum(xv,axis=1)) - \
             K.sum(K.square(V_embedding)* K.square(K.expand_dims(new_input,2)),axis=1)
         )
         y_FM #/= batch_size
         return y_FM

    def _build(self):
        int_dim = self.config['int_features']
        float_dim = self.config['float_features']
        feature_sizes = self.config['int_feature_sizes']
        embed_size = self.config['embed_size']
        all_dim = int_dim + float_dim
        self.int_dim = int_dim
        self.float_dim = float_dim
        self.feature_sizes = feature_sizes
        self.all_dim = all_dim

        input = Input(shape=(all_dim,),name="Input")
        input_embedding = Lambda(self.build_embedding_input)(input)
        new_input = Lambda(self.build_new_input)(input)

        total_feature_size = sum(feature_sizes) + float_dim
        # Add Embedding For all features
        # [None, all_dim, emb_size]
        V_embedding = Embedding(total_feature_size,embed_size)(input_embedding)
        # [None, all_dim, 1]
        x_embedding = Embedding(total_feature_size, 1)(input_embedding)

        # FM Part
        y_FM = Lambda(function=self.FM_part)([x_embedding,V_embedding,new_input])
        #y_FM = self.FM_part(x_embedding, V_embedding, new_input)

        # DNN Part
        flatten = Flatten()(V_embedding)

        XF = Dropout(self.config['input_dropout'])(flatten)
        """
        for i in range(self.config['hidden_layers']):
            #GLU
            dense = Dense(self.config['hidden_units'])(XF)
            gate = Dense(self.config['hidden_units'],activation='sigmoid')(XF)
            XF = multiply([dense,gate]) #以上三步构成了所谓的GLU激活函数
            #Normal dense
            #input = Dense(self.config['hidden_units'],activation=self.config['hidden_activation'])(input)
            #input = Dropout(self.config['hidden_dropout'])(input)
        """
        XF = Dense(100,activation="relu")(XF)
        y_DNN = Dense(1,activation="relu")(XF)
        y_sum = add([y_DNN,y_FM])
        #predict = y_sum
        #y_sum = y_FM
        predict = Dense(1,activation='sigmoid')(y_sum)
        model = Model(inputs=input,outputs=predict)
        lr = self.config['learning_rate']
        print("learning_rate=%f"%(lr))
        optimizer = Adam(lr)
        #model.compile(loss = "binary_crossentropy",
        model.compile(loss = focal_loss, #loss="binary_crossentropy",
                      metrics=['accuracy',auc],
                      #metrics=['accuracy'],
                      optimizer=self.config['optimizer'])
        self.model = model
        model.summary()

    def fit(self,X, y, validate=True):
        X = np.nan_to_num(X)
        self.config['nb_epoch'] = 1000
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
        X = np.nan_to_num(X)
        return self.model.predict(X)

## regression with Keras' deep neural network
class DeepFMLearner(BaseLearner):
    name = "reg_deepfm"

    param_space = {
        "int_features": 6,
        "float_features":32,
        'int_feature_sizes': [364779,769,4228,501,957,24],
        'embed_size': hp.choice('embed_size',[32,64]),
        "input_dropout": hp.quniform("input_dropout", 0, 0.2, 0.05),
        "hidden_layers": hp.quniform("hidden_layers", 1, 3, 1),
        "hidden_units": hp.quniform("hidden_units", 32, 128, 32),
        "hidden_activation": hp.choice("hidden_activation", ["relu", "elu"]),
        "hidden_dropout": hp.quniform("hidden_dropout", 0, 0.5, 0.05),
        "batch_norm": hp.choice("batch_norm", ["before_act", "after_act", "no"]),
        "optimizer": hp.choice("optimizer", ["sgd", "adam", "adadelta", "rmsprop"]),
        "batch_size": hp.quniform("batch_size", 1000, 10000, 2000),
        "nb_epoch": hp.quniform("nb_epoch", 1, 20, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01)),
    }

    def create_model(self,params):
        self.params = params

        return DeepFMModel(params)

