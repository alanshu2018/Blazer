#coding:utf-8

import numpy as np
from Learner import BaseLearner
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Input,Embedding, Dense, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, GlobalMaxPooling1D, Lambda, Flatten
from keras.datasets import imdb
from keras import backend as K
from keras.models import Model
from keras.layers.merge import multiply, add
from keras.optimizers import Adam,SGD,RMSprop, Nadam


from keras.callbacks import Callback, EarlyStopping
from keras.utils.vis_utils import plot_model

from keras.callbacks import TensorBoard
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

focal_loss = lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2
"""
# Kaiming He's focal loss
def focal_loss(y_true, y_pred, e=0.1,nb_classes=2):
    alpha= 0.5 #0.25
    gamma = 2
    return -y_true*alpha*K.pow(1-y_pred,gamma)*K.log(y_pred) - (1-y_true)*(1-alpha)*K.pow(y_pred,gamma)*K.log(1-y_pred)
"""
#callbacks = [EarlyStopping(monitor='val_acc',patience=3,verbose=0,mode='max')]
callbacks = [
    EarlyStopping(monitor='val_acc',patience=3,verbose=0,mode='max'),
    TensorBoard(log_dir='./tmp/log'),
]

class DeepFMRankModel(object):
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
         w_embedding, V_embedding, x_input = X
         # x_input_expand: [None, all_dim, 1]
         x_input_expand = K.expand_dims(x_input, -1)
         # x_embeding: [None, all_dim, 1]
         y_FM = K.sum(w_embedding * x_input_expand, axis=[1,2])
         # x_input_expand: [None, all_dim, 1]
         # V_embedding [None, all_dim, embed_size]
         # xv [None, all_dim, embed_size]
         print("y_FM={}".format(y_FM))
         xv = V_embedding * x_input_expand
         y_FM += 0.5 * K.sum(
             K.square(K.sum(xv,axis=1)) - \
             K.sum(K.square(V_embedding)* K.square(x_input_expand),axis=1),
             axis=1,
         )
         y_FM = K.reshape(y_FM,(-1,1))
         print("y_FM={}".format(y_FM))
         #y_FM /= batch_size
         return y_FM

    def DNN_part(self,X):
        w_embedding, V_embedding, x_input = X
        # DNN Part
        x_input_expand = K.expand_dims(x_input,-1)
        V = multiply([V_embedding,x_input_expand])
        flatten = Flatten()(V)

        XF = Dropout(self.config['input_dropout'])(flatten)
        for i in range(self.config['hidden_layers']):
            #GLU
            dense = Dense(self.config['hidden_units'])(XF)
            gate = Dense(self.config['hidden_units'],activation='sigmoid')(XF)
            #XF = multiply([dense,gate]) #以上三步构成了所谓的GLU激活函数
            XF = dense
            #Normal dense
            #input = Dense(self.config['hidden_units'],activation=self.config['hidden_activation'])(input)
            #input = Dropout(self.config['hidden_dropout'])(input)
        XF = Dense(100,activation="relu")(XF)
        y_DNN = Dense(1,activation="relu")(XF)
        print("y_DNN={}".format(y_DNN))

        return y_DNN

    def build_w_embedding(self,input):
        w_embeds = []
        for i in range(self.int_dim):
            emb = Embedding(self.feature_sizes[i],1)(input[:,i])
            w_embeds.append(K.expand_dims(emb,1))

        for i in range(self.float_dim):
            emb = Embedding(1,1)(K.zeros_like(input[:,i+self.int_dim]))
            w_embeds.append(K.expand_dims(emb,1))

        w_embeds = K.concatenate(w_embeds,axis=1)
        return w_embeds

    def build_V_embedding(self,input):
        V_embeds = []
        for i in range(self.int_dim):
            emb = Embedding(self.feature_sizes[i],self.embed_size)(input[:,i])
            V_embeds.append(K.expand_dims(emb,1))

        for i in range(self.float_dim):
            emb = Embedding(1,self.embed_size)(K.zeros_like(input[:,i+self.int_dim]))
            V_embeds.append(K.expand_dims(emb,1))

        #print(V_embeds)
        V_embeds = K.concatenate(V_embeds,axis=1)
        print(V_embeds)
        return V_embeds

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
        self.embed_size = embed_size
        total_feature_size = sum(feature_sizes) + float_dim

        input = Input(shape=(all_dim,),name="Input")

        new_input = Lambda(self.build_new_input)(input)

        new_input = BatchNormalization()(new_input)

        # Add Embedding For all features
        # [None, all_dim, 1]
        w_embedding = Lambda(self.build_w_embedding)(input)
        # [None, all_dim, emb_size]
        V_embedding = Lambda(self.build_V_embedding)(input)

        # FM Part
        y_FM = Lambda(function=self.FM_part,output_shape=(1,))([w_embedding,V_embedding,new_input])
        #y_FM = self.FM_part(x_embedding, V_embedding, new_input)
        y_DNN = Lambda(function=self.DNN_part,output_shape=(1,))([w_embedding,V_embedding,new_input])

        y_sum = add([y_DNN,y_FM])
        #predict = y_sum
        #y_sum = y_FM
        predict = Dense(1,activation='relu')(y_sum)
        self.predict_model = Model(inputs=input,outputs=predict)
        self.predict_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy',auc])
        self.predict_model.summary()

        ##################################################################
        # Build Rank model
        ##################################################################
        lr = self.config['learning_rate']
        print("learning_rate=%f"%(lr))
        optimizer = Nadam(lr)
        #optimizer = SGD(lr,momentum=0.95,decay=0.99)

        pos_input = Input(shape=(self.all_dim,),name="Pos_Inp")
        neg_input = Input(shape=(self.all_dim,),name="Neg_Inp")

        pos_score = self.predict_model(pos_input)
        neg_score = self.predict_model(neg_input)

        margin = 0.6
        loss = Lambda(lambda x : K.relu(margin - x[0]+x[1]),name="loss")([pos_score,neg_score])
        #loss = Lambda(lambda x : K.sigmoid(-x[0] + x[1]),name="loss")([pos_score,neg_score])
        """
        train_model = Model(inputs=[pos_input,neg_input],outputs=[pos_score,neg_score,loss])
        train_model.compile(optimizer=optimizer,
                            loss=['binary_crossentropy','binary_crossentropy',lambda y_true, y_pred: y_pred],
                            loss_weights=[5.0,1.0,10.0],
                            metrics={'loss':'binary_crossentropy'})
        """
        train_model = Model(inputs=[pos_input,neg_input],outputs=loss)
        train_model.compile(optimizer=optimizer,
                            loss=lambda y_true, y_pred: y_pred,
                            )

        self.model = train_model
        self.model.summary()

        plot_model(self.model,"deepfmrank_train_model.png")
        plot_model(self.predict_model,"deepfmrank_predict_model.png")

    def fit(self,X, y, validate=True,weight=None):
        X = np.nan_to_num(X)
        X[self.int_dim:] = np.log(np.log(X[self.int_dim:] + 1.0)+1.0)
        pos_inds = y[y>0]
        neg_inds = y[y<=0]

        Xpos = X[pos_inds]
        Xneg = X[neg_inds]

        num_pos = len(Xpos)
        num_neg = len(Xneg)
        pos_arr = range(num_pos)
        neg_arr = range(num_neg)
        print("num_pos={}, num_neg={}".format(num_pos,num_neg))
        epochs = 10
        batch_size = self.config['batch_size']

        pos_labels = np.ones((batch_size,))
        neg_labels = np.zeros((batch_size,))
        rand_labels = np.random.random_sample((batch_size,))
        for n in range(epochs):
            # get a batch_size of pos_examples
            sel_pos_ind = np.random.choice(pos_arr,size=(batch_size,),replace=True)
            sel_neg_ind = np.random.choice(neg_arr,size=(batch_size,),replace=True)
            sel_pos = Xpos[sel_pos_ind]
            sel_neg = Xneg[sel_neg_ind]

            if validate:
                #self.model.fit([sel_pos,sel_neg],[pos_labels,neg_labels,rand_labels],
                self.model.fit([sel_pos,sel_neg],rand_labels,
                       batch_size=self.config['batch_size'],
                       epochs=self.config['nb_epoch'],
                       callbacks = callbacks,
                       verbose=1,validation_split=0.1)
            else:
                #self.model.fit([sel_pos,sel_neg],[pos_labels,neg_labels,rand_labels],
                self.model.fit([sel_pos,sel_neg],rand_labels,
                           batch_size=self.config['batch_size'],
                           epochs=self.config['nb_epoch'],
                           callbacks = callbacks,
                           verbose=1)

            print("test_pos={}".format(self.predict(sel_pos[:5])))
            print("test_neg={}".format(self.predict(sel_neg[:5])))

    def predict(self,X):
        X = np.nan_to_num(X)
        X[self.int_dim:] = np.log(np.log(X[self.int_dim:] + 1.0)+1.0)
        #return self.model.predict(X)
        return self.predict_model.predict(X)

## regression with Keras' deep neural network
class DeepFMRankLearner(BaseLearner):
    name = "reg_deepfm_rank"

    param_space = {
        "int_features": 6,
        "float_features":29,
        'int_feature_sizes': [364779,769,4228,501,957,24],
        'embed_size': hp.choice('embed_size',[32,64,128]),
        "input_dropout": hp.quniform("input_dropout", 0, 0.2, 0.05),
        "hidden_layers": hp.quniform("hidden_layers", 1, 3, 1),
        "hidden_units": hp.quniform("hidden_units", 32, 128, 32),
        "hidden_activation": hp.choice("hidden_activation", ["relu", "elu"]),
        "hidden_dropout": hp.quniform("hidden_dropout", 0, 0.5, 0.05),
        "batch_norm": hp.choice("batch_norm", ["before_act", "after_act", "no"]),
        "optimizer": hp.choice("optimizer", ["adam", "adadelta", "rmsprop"]),
        "batch_size": hp.quniform("batch_size", 10000, 20000, 2000),
        "nb_epoch": hp.quniform("nb_epoch", 1, 20, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01)),
    }

    def create_model(self,params):
        self.params = params

        return DeepFMRankModel(params)

