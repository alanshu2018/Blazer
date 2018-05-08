#coding:utf-8

import tensorflow as tf

import sys
from model import *
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from DataLoader import DataLoader
import gc
from LRTunner import Tunner
from hyperopt import fmin, hp, tpe
import hyperopt

ver="v19"

click_means=np.array([0]*8+[1.866671e+07,5.083459e+04,2.351688e+07,1.809987e+08,4.543603e+06,
             1.215442e+05,2.213557e+06,9.366850e+05,1.703376e+07,1.928687e+05,
             3.039419e+06,3.092410e+02,5.026361e+03,1.170573e+06,2.202677e+07,
             ]+[0.0] *30)
click_std=np.array([1.0]*8 +[ 1.147628e+07, 1.801862e+05, 2.076854e+07, 4.308380e+07, 3.937021e+06,
            1.535289e+05, 2.650926e+06, 6.892881e+05, 1.136252e+07, 2.888697e+05,
            3.759831e+06, 1.343773e+03, 2.044271e+04, 1.112249e+06, 2.028386e+07,
            ]+[1.0]*30)

class Config(object):
    feature_names = [
        'ip','app','device','channel','os','day','hour','hour_4'
    ]
    predictors = [
        'ip','app','device','channel','os','day','hour','hour_4',
        'c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11','c12','c13','c14','c15',
        'r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13','r14','r15',
        'h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15',
              ]

    """
    feature_names = ["ip","app","device","os","channel","day","wday","hour","hour_4"] #,"min_6"]
    feature_names_float = [
        "ip_os_dev_count","ip_app_dev_count","ip_app_count","ip_app_wday_hour_count", #4
        "ip_app_day_hour_count","ip_app_hour_count","ip_dev_count", #3
        "qty_wday_hour","qty_day_hour","qty_hour","app_os_count", #4
        "app_count","app_os_dev_count","app_dev_count","ip_app_os_count" #4
    ]
    """
    feature_sizes = [364778,768,4227,1000,956,10,24,6]
    #feature_names = ["app","device","os","channel","day","wday","hour","hour_4"] #,"min_6"]
    #feature_sizes = [768,4227,956,500,10,7,24,6]
    #feature_names = ["day"] #,"min_6"]
    #feature_sizes = [10]
    #feature_names_float = ["app_count"]

    int_features = 8 #len(feature_sizes)
    float_features = 45#len(feature_names_float)

    feature_num = int_features + float_features
    all_feature_names = predictors #feature_names + feature_names_float
    embed_size = 50
    #vocab_size = sum(feature_sizes)
    hidden_size = 256

    n_class = 2
    lr = 1e-4
    n_epochs = 2
    n_steps = -1
    batch_size = 100000
    dropout = 0.5

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high, dtype=tf.float32)

class DeepFMModel(Model):
    name = "deepfm_{}".format(ver)
    #config = Config()

    def __init__(self,config):
        self.config = config
        tf.reset_default_graph()
        self.build()

    def add_placeholders(self):
        #ip, app, os, device, os, channel, day, wday, hour, hour/4, min/6
        self.X = tf.placeholder(dtype=tf.float32, shape=[None,self.config.feature_num],name="X")
        self.Y = tf.placeholder(dtype=tf.int32,shape=[None],name="Y")
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def add_embedding(self,X):
        with tf.variable_scope("embed") as scope:
            self.embeddings = []
            embeds = []
            for (idx, size) in enumerate(self.config.feature_sizes):
                embeddings = tf.Variable(
                    #tf.random_uniform([size,self.config.embed_size+1],-1.0,1.0),
                    xavier_init(size,self.config.embed_size+1),
                    name="embeddings_{}".format(idx)
                )
                self.embeddings.append(embeddings)
                embed = tf.nn.embedding_lookup(embeddings,tf.slice(X,[0,idx],[-1,1]))
                #embed = tf.reshape(embed,[-1, self.config.embed_size])
                embeds.append(embed)
            embeds = tf.concat(embeds,axis=1)
        return embeds

    def add_prediction_op(self):
        X1 = tf.cast(tf.slice(self.X,begin=[0,0],size=[-1,self.config.int_features]),tf.int32)
        X2 = tf.slice(self.X,begin=[0,self.config.int_features],size=[-1,self.config.float_features])
        #print(tf.shape(X1))
        #print(tf.shape(X2))
        X_First = self.add_embedding(X1)
        # Linear weight for feature: [None,int_feature, 2]
        X_First0 = tf.cast(tf.slice(X_First,begin=[0,0,0],size=[-1,-1,1]),tf.float32)

        # Vector for feature
        #[None, int_feature, embed_size]
        X_First1 = tf.cast(tf.slice(X_First,begin=[0,0,1],size=[-1,-1,self.config.embed_size]),tf.float32)
        #X_First1 = tf.reshape(X_First1,[-1, self.config.int_features*self.config.embed_size])

        #X_Second = tf.layers.batch_normalization(tf.cast(X2,tf.float32))
        #Float feature
        #[None,float_feature]
        X_Second = tf.cast(X2,tf.float32)
        #X1 = tf.cast(X1,tf.float32)

        #X = tf.concat([X_First,X_Second],axis=-1)
        #X = tf.concat([X1,X_Second],axis=-1)
        #X_shape = tf.shape(X)
        #X = X_First

        with tf.variable_scope("predict") as scope:
            # Linear term
            W1 = tf.Variable(
                xavier_init(self.config.float_features,
                            1),
                #tf.random_uniform([self.config.int_features*self.config.embed_size + self.config.float_features,
                #                   self.config.hidden_size])*0.001,
                name = "W1")
            b1 = tf.Variable(
                tf.zeros(shape=1), name = "b1")

            X_First0 = tf.reshape(X_First0,[-1,self.config.int_features,1])
            self.linear_terms = tf.matmul(X_Second,W1) + b1 + tf.reduce_sum(X_First0,axis=[1])

            # Interactions
            # Embedding for float features
            V_f = tf.Variable(
                xavier_init(self.config.float_features,self.config.embed_size),
                name="V_f"
            )

            feature_num = self.config.int_features + self.config.float_features
            X_shape = tf.shape(X_First0)
            #[None,int_feature]
            X_ones = tf.ones((X_shape[0],X_shape[1]),dtype=tf.float32)
            # [None, float_feature, embed_size]
            #VV = tf.matmul(tf.ones((X_shape[0],1,self.config.float_features),dtype=tf.float32),V_f_3d)
            #VV = tf.reshape(tf.tile(V_f,X_shape[0]),[-1,self.config.float_features,self.config.embed_size])
            VV = tf.ones((X_shape[0],self.config.float_features,self.config.embed_size)) * V_f
            # X_Second: [None, float_feature]
            #[None, int_feature + float_feature]
            Xf = tf.concat([X_ones, X_Second],axis=1)
            # X_First1 : [None, int_feature, embed_size]
            # VV: [None, float_feature, embed_size]
            # V: [None,int_feature + float_feature,embed_size]
            # Xf: [None, int_feature + float_feature]
            V = tf.concat([X_First1, VV],axis=1)
            #self.interaction_terms = tf.multiply(0.5 , tf.reduce_mean(
            #                    tf.pow(V*Xf,2) - tf.pow(Xf,2) *tf.pow(V,2),
            #                    axis=[1,2]
            #                    ))
            # XV: [None,int_feature + float_feature,embed_size]
            XV = V * tf.reshape(Xf,[-1,feature_num,1])
            XV_Squared = XV * XV
            XV_Sum = tf.reduce_sum(XV,axis=[1])
            XV_Squared_Sum = tf.reduce_sum(XV_Squared,axis=[1])
            self.interaction_terms = tf.reshape(0.5 * tf.reduce_sum(XV_Sum * XV_Sum - XV_Squared_Sum,axis=[1]),(-1,1))
            #self.interaction_terms = 0.5 * tf.reduce_sum(tf.matmul(XV,tf.transpose(XV,[0,2,1])),axis=[1,2])

            self.y_fm = self.linear_terms + self.interaction_terms

            # DNN parts
            y_embedding = tf.reshape(V,[-1, feature_num * self.config.embed_size])

            W2 = tf.Variable(
                xavier_init(feature_num * self.config.embed_size,
                        self.config.hidden_size),
                name = "W2")
            b2 = tf.Variable(tf.zeros(shape=(self.config.hidden_size)),name="b2")

            W3 = tf.Variable(
                xavier_init(self.config.hidden_size,1),
                name = "W3")
            b3 = tf.Variable(tf.zeros(shape=(1)),name="b3")

            # swish
            #y_embedding = tf.nn.dropout(y_embedding, self.dropout_placeholder)
            h = tf.matmul(y_embedding,W2) + b2
            h = tf.nn.sigmoid(h) * h
            #h = tf.nn.relu(h)
            h_drop = tf.nn.dropout(h, self.dropout_placeholder)

            h = h_drop
            h = tf.matmul(h,W3) + b3
            h = tf.nn.sigmoid(h) * h

            self.y_dnn = h

            self.y_out = self.y_fm + self.y_dnn
            self.pred_sigmoid = tf.nn.sigmoid(self.y_out)

            return self.y_out

    def add_loss_op(self, pred):
        with tf.variable_scope("loss") as scope:
            shape = tf.shape(pred)
            #Y = tf.one_hot(self.Y,depth=self.config.n_class)
            """
            Y_pred = tf.nn.softmax(pred)
            Y_pred = tf.slice(Y_pred,begin=[0,1],size=[-1,1],name="y_pred")
            t1 = tf.maximum(0.0,0.9-Y_pred)
            t2 = tf.maximum(0.0,Y_pred - 0.1)
            YY = tf.reshape(tf.cast(self.Y,tf.float32),tf.shape(t1))
            loss1 = YY * t1 * t1 + 0.25 *(1-YY)* t2 * t2
            loss1 = tf.reduce_mean(loss1)
            loss =loss1
            """
            Y_shaped = tf.cast(tf.reshape(self.Y,(shape[0],1)),tf.float32)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_shaped,logits = pred)
            #loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits= pred)
            loss = tf.reduce_mean(loss)
            _, auc = tf.metrics.auc(Y_shaped,self.pred_sigmoid)
            y_pred = tf.cast(tf.reshape(tf.cast(self.pred_sigmoid,tf.int32),(shape[0],1)),tf.float32)
            acc = tf.reduce_mean(tf.cast(tf.equal(Y_shaped,y_pred),tf.float32))

            tf.summary.scalar('accuracy',acc)
            tf.summary.scalar('auc',auc)
            tf.summary.scalar('loss',loss)
            return loss, auc, acc

    def add_training_op(self, loss):
        #lr_init, lr_fin = self.config.lr,self.config.lr * 0.1
        #exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
        #steps = int(1e8 / self.config.batch_size) * self.config.n_epochs
        #lr_decay = exp_decay(lr_init, lr_fin, steps)

        global_step = tf.Variable(0, trainable=False)
        #lr_decay = tf.train.exponential_decay(self.config.lr, global_step, 1, 0.9999)
        #optimizer = tf.train.AdamOptimizer(self.config.lr)
        lr_decay = self.config.lr

        optimizer = tf.train.AdamOptimizer(lr_decay)
        l,auc,acc = loss
        train_op = optimizer.minimize(l,global_step= global_step)
        return train_op

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        #inputs_batch[:,:self.config.int_features] += self.config.feature_sizes_add
        feed_dict = {
            self.X: inputs_batch,
        }
        if labels_batch is not None:
            feed_dict[self.Y] = labels_batch
        else:
            raise Exception("Labels_path is None")
        return feed_dict

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch)
        feed[self.dropout_placeholder] = self.config.dropout
        _, loss, train_summary = sess.run([self.train_op, self.loss, self.merge_summary], feed_dict = feed)
        #auc = roc_auc_score(labels_batch,pred1[:,0])
        #print("pred={}".format(pred[:10]))
        return loss,train_summary

    def evaluate(self, sess, dev_data):
        ce_loss = 0.0
        #auc = 0.0
        acc = 0.0
        total = 0
        predicted = [] #np.zeros((len(dev_data)))
        ground_truth = [] #np.zeros(len(dev_data))
        start = 0
        for (dev_data_x,dev_data_y) in dev_data.get_minibatch(batch_size=20000):
            #dev_data_x, dev_data_y = split_x_y(dev_data)
            feed = self.create_feed_dict(dev_data_x,dev_data_y)
            feed[self.dropout_placeholder] = 1.0
            loss,pred_sigmoid = sess.run([self.loss,self.pred_sigmoid], feed_dict=feed)
            closs, aloss,accloss = loss
            total += 1
            if total % 20 ==0:
                print("Evaluate: total={}".format(total))
            ce_loss += closs
            #auc += aloss
            acc += accloss
            predicted.extend(pred_sigmoid.tolist())
            ground_truth.extend(dev_data_y)
        auc = roc_auc_score(ground_truth,predicted)
        #print("sum of predicted={}".format(sum(predicted)))
        #print(predicted[:50])
        return ce_loss/float(total),auc, acc/float(total) #[0],auc

    def predict(self, sess, test_data):
        preds = []
        labels = []
        for X, y in test_data.get_minibatch(batch_size=self.config.batch_size*4,do_shuffle=False):
            feed = self.create_feed_dict(X)
            feed[self.dropout_placeholder] = 1.0
            pred = sess.run(self.pred_sigmoid,feed_dict= feed)
            preds.extend(pred)
            labels.extend(y)
        return preds, labels

    def load_best_model(self, sess, saver):
        checkpoint_dir = "newdata/model_{}".format(self.name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Load Model from path:{}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

    def fit(self, sess, train_writer, saver, train_data, dev_data, epochs=None):
        best_auc = 0
        best_step = 0
        step = 0
        if epochs is None:
            epochs = self.config.n_epochs

        self.merge_summary = tf.summary.merge_all()
        print("Total epochs={}".format(epochs))
        for i, (train_x, train_y) in enumerate(train_data.get_minibatch(epochs=epochs)):
            step += 1
            all_loss,train_summary = self.train_on_batch(sess, train_x, train_y)
            loss,auc, acc = all_loss
            train_writer.add_summary(train_summary,step)

            #prog.update(i+1, [("train_loss",loss),("train_auc",auc)])
            if i % 10 == 0:
                print("Step:{}, loss:{},auc:{},acc:{}".format(i, loss, auc,acc))

            if i > 0 and i % 30 == 0:
                dev_loss,dev_auc,dev_acc = self.evaluate(sess,dev_data)
                #prog.update(i+1, [("dev_loss",dev_loss),("dev_auc",dev_auc)])
                print("==>Evaluate: Step:{}, loss:{},auc:{},acc:{}".format(i, dev_loss, dev_auc,dev_acc))
                if dev_auc > best_auc:
                    best_auc = dev_auc
                    best_step = i
                    print("***Best model: loss:{}, auc:{},acc:{}".format(dev_loss, dev_auc,dev_acc))
                    save_path = saver.save(sess,"./newdata/model_{}/{}".format(self.name,self.name))
                    print("Save in path = {}".format(save_path))
                elif i - best_step > 500:
                    print("***Early Stopping:step={},best_step={}, best_auc={}".format(i,best_step,best_auc))
                    return best_auc, best_step

            if self.config.n_steps >0 and i >= self.config.n_steps:
                print("***Max step reaches:step={},max={}".format(i,self.config.n_steps))
                break


        print("***Finish epochs: Epochs:{}, Step:{}, auc:{}".format(epochs, best_step, best_auc))
        return dev_auc, step
        #return loss, auc

"""
train_data_loader = DataLoader(files=[
    ("data/dev1_feature_norm.simple.v2.csv",2000868),
])

test_data_loader = DataLoader(files=[
    ("data/dev1_feature_norm.simple.v2.csv",2000868),
])

config = Config()
model = LogisticRegressionModel(config)
saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    model.fit(sess,saver,train_data_loader,test_data_loader)
    #save_path = saver.save(sess,"./data/model_{}/{}".format("lr_dnn","lr_dnn"))

import sys
sys.exit(-1)
"""
class LRTunner(Tunner):
    def __init__(self, train_data, dev1_data,dev2_data):
        self.train_data = train_data
        self.dev1_data = dev1_data
        self.dev2_data = dev2_data

    def get_space(self):
        # Searching space
        space = {
            'learning_rate': hp.choice("learning_rate", [1e-3]), #[0.05,0.01,0.1]),
            #'learning_rate': hp.choice("learning_rate", [1e-3]), #[0.05,0.01,0.1]),
            'embed_dim':hp.choice("embed_dim",[50,80,100]),
            #'embed_dim':hp.choice("embed_dim",[20]),
            'epochs':hp.choice("epochs",[20]),
            'dropout': hp.choice("dropout", [0.1, 0.3,0.5]),
            #'dropout': hp.choice("dropout", [0.5,0.6]),
        }
        return space

    def build(self, args):
        # args to config
        print("Build with args:{}".format(args))
        config = Config()
        config.lr = args['learning_rate']
        config.embed_size = args['embed_dim']
        config.n_epochs = args['epochs']
        config.n_steps = args.get('max_steps',-1)
        config.dropout = args['dropout']

        model = DeepFMModel(config)
        return model, config

    def train_and_evaluate(self,model,config):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter("{}_log".format(model.name),sess.graph)

            sess.run(init)
            sess.run(tf.local_variables_initializer())
            auc, step = model.fit(sess,train_writer, saver,self.train_data,self.dev1_data)


        auc_error = 1 - auc
        return auc_error, step

    def train(self,sess, saver, model,train_data,dev_data,epochs=None):
        auc, step = model.fit(sess,saver,train_data,dev_data,epochs)
        return model, auc, step

    def load_best_model(self,sess, saver, model):
        model.load_best_model(sess, saver)
        return model

    def predict_and_save(self,sess,name, model, test_data):
        print("Predicting for name:{}".format(name))
        sub = pd.DataFrame()
        predicted, labels = model.predict(sess,test_data)
        sub['click_id'] = map(int,labels) #.astype('int')
        sub['is_attributed'] = predicted
        print("writing {}......".format(name))
        sub.to_csv(name,index=False)
        del sub
        gc.collect()

    def train_and_predict(self,name,model,train_data,dev_data,test_data,output_dev=True,output_test=True):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            sess.run(tf.local_variables_initializer())
            model, auc, step = self.train(sess, saver, model, train_data, dev_data)

            #predict on dev for ensembling
            if output_dev:
                file = "dev_{}.csv".formate(name)
                self.predict_and_save(sess,file,model,dev_data)

            if output_test:
                file = "test_{}.csv".formate(name)
                self.predict_and_save(sess,file,model,test_data)

        return auc, step

    def predict_submission(self, session, model):
         # predict
        print ("Load test data")
        test_data = pd.read_hdf("data/test.hdf","data")
        #print(test_data.head())
        sub = pd.DataFrame()
        sub['click_id'] = test_data['click_id'] #.astype('int')
        print("Predicting...")
        sub['is_attributed'] = model.predict(session,test_data)
        file ='sub_LR_emd_{}_hidden_{}_lr_{}_epoch_{}.csv'.format(
            model.config.embed_size,
            model.config.hidden_size,
            model.config.lr,
            model.config.n_epochs,
        )
        print("writing {}......".format(file))
        sub.to_csv(file,index=False)

import time
import pandas as pd
import sys

def main(debug=True):
    print("Building model...")
    start = time.time()

    train_data_loader = DataLoader(files=[
        #("data/dev1_feature_norm.simple.{}.csv".format(ver),2000868),
        #("data/dev2_feature_norm.simple.{}.csv".format(ver),1999132),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv".format(ver),180903890),
        ("newdata/train_small_{}.csv".format(ver),50000000),
    ])
    train_data_loader1 = DataLoader(files=[
        #("data/dev2_feature_norm.simple.{}.csv".format(ver),1999132),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv".format(ver),180903890),
        ("newdata/train_small_{}.csv".format(ver),50000000),
        ("newdata/dev1_{}.csv".format(ver),2000000),
        #("data/train_feature_norm.simple.small.{}.csv".format(ver),30000000),
    ])
    train_data_loader2 = DataLoader(files=[
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.small.{}.csv".format(ver),30000000),
        ("newdata/train_small_{}.csv".format(ver),50000000),
        ("newdata/dev1_{}.csv".format(ver),2000000),
        ("newdata/dev2_{}.csv".format(ver),2000000),
    ])

    dev1_data_loader = DataLoader(files=[
        #("data/dev1_feature_norm.simple.{}.csv".format(ver),2000868),
        ("newdata/dev1_{}.csv".format(ver),2000000),
        #("data/dev2_feature_norm.simple.{}.csv".format(ver),1999132),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv".format(ver),180903890),
    ])

    dev2_data_loader = DataLoader(files=[
        #("data/dev1_feature_norm.simple.{}.csv".format(ver),2000868),
        ("newdata/dev2_{}.csv".format(ver),2000000),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv".format(ver),180903890),
    ])

    dev3_data_loader = DataLoader(files=[
        ("newdata/dev1_{}.csv".format(ver),2000000),
        ("newdata/dev2_{}.csv".format(ver),2000000),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv".format(ver),180903890),
    ])

    test_data_loader = DataLoader(files=[
        #("data/dev1_feature_norm.simple.{}.csv".format(ver),2000868),
        #("data/dev2_feature_norm.simple.{}.csv".format(ver),1999132),
        #("data/test_feature_norm.simple.{}.csv".format(ver),18790469),
        ("newdata/test_{}.csv".format(ver),18790469),
        #("data/train_feature_norm.simple.{}.csv",180903890),
    ])

    if True:
        # Train on small train, and evaluate on dev1
        tunner = LRTunner(train_data_loader, dev1_data_loader, dev2_data_loader)
        best_sln = tunner.tune()
        print("best_sln={}".format(best_sln))

    if False:
        args = {
            #'learning_rate': hp.choice("learning_rate", [1e-3,5e-4,1e-4]), #[0.05,0.01,0.1]),
            'learning_rate': 1e-3, #hp.choice("learning_rate", [1e-3,1e-4]), #[0.05,0.01,0.1]),
            #'embed_dim':hp.choice("embed_dim",[20,50]),
            'embed_dim':20, #hp.choice("embed_dim",[20]),
            'epochs': 20, #hp.choice("epochs",[10]),
            #'dropout': hp.choice("dropout", [0.1, 0.3,0.5]),
            'dropout': 0.5, #hp.choice("dropout", [0.5]),
            'max_steps': 8000,
        }
        arg_str="lr_{}_emb_{}_epochs_{}_dropout_{}_steps_{}".format(
            args["learning_rate"],
            args["embed_dim"],
            args["epochs"],
            args["dropout"],
            args["max_steps"],
        )
        date_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        data_tag = "%s_%s"%(ver,date_str)
        print("Build with args:{}".format(args))
        tunner = LRTunner(train_data_loader, dev1_data_loader, dev2_data_loader)
        model, config = tunner.build(args)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            sess.run(tf.local_variables_initializer())

            # Train on train + dev1, predict on dev2
            #tunner.train(sess,saver,model,train_data_loader1,dev2_data_loader)
            model, auc, step = tunner.train(sess,saver,model,train_data_loader,dev2_data_loader)
            print("***Train First stage Finished")
            model = tunner.load_best_model(sess,saver,model)
            tunner.predict_and_save(sess,"dev_" +data_tag + "_" + arg_str,model,dev2_data_loader)

            # Train on all the dev data again, and generate the submission
            model, auc, step = tunner.train(sess,saver,model,dev3_data_loader,dev2_data_loader,epochs=2)
            print("***Train Second stage Finished")
            model = tunner.load_best_model(sess,saver,model)
            tunner.predict_and_save(sess,"sub_" + data_tag + "_" + arg_str,model,test_data_loader)

    end = time.time()
    print("Cost {} seconds".format(end - start))


if __name__ == "__main__":
    main()
