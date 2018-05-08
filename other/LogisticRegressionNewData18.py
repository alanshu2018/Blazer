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

ver="v18"

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
    hidden_size = 512

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

class LogisticRegressionModel(Model):
    name = "lr_dnn_{}".format(ver)
    #config = Config()

    def __init__(self,config):
        self.config = config
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
                    tf.random_uniform([size,self.config.embed_size],-1.0,1.0),
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
        X_First = tf.reshape(X_First,[-1, self.config.int_features*self.config.embed_size])

        #X_Second = tf.layers.batch_normalization(tf.cast(X2,tf.float32))

        X_Second = tf.cast(X2,tf.float32)
        #X1 = tf.cast(X1,tf.float32)

        X = tf.concat([X_First,X_Second],axis=-1)
        #X = tf.concat([X1,X_Second],axis=-1)
        #X_shape = tf.shape(X)
        #X = X_First

        with tf.variable_scope("predict") as scope:
            W1 = tf.Variable(
                xavier_init(self.config.int_features * self.config.embed_size + self.config.float_features,
                            self.config.hidden_size),
                #tf.random_uniform([self.config.int_features*self.config.embed_size + self.config.float_features,
                #                   self.config.hidden_size])*0.001,
                name = "W1")
            b1 = tf.zeros(shape=self.config.hidden_size, name = "b1")
            W2 = tf.Variable(
                xavier_init(self.config.hidden_size,
                            self.config.n_class),
                #tf.random_uniform([self.config.hidden_size,
                #                     self.config.n_class])*0.01,
                name = "W2")
            b2 = tf.zeros(shape=(self.config.n_class),name="b2")

            # swish
            #X = tf.nn.dropout(X, self.dropout_placeholder)
            h = tf.matmul(X,W1) + b1
            #h = tf.nn.sigmoid(h) * h
            h = tf.nn.relu(h)
            h_drop = tf.nn.dropout(h, self.dropout_placeholder)

            pred = tf.matmul(h_drop, W2) + b2
            #pred = tf.nn.softmax(pred)
            #self.pred1 = tf.nn.sigmoid(pred)
            self.pred_softmax = tf.nn.softmax(pred)

            return pred

    def add_loss_op(self, pred):
        with tf.variable_scope("loss") as scope:
            shape = tf.shape(pred)
            Y = tf.one_hot(self.Y,depth=self.config.n_class)
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
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits= pred)
            loss = tf.reduce_mean(loss)
            _, auc = tf.metrics.auc(Y,self.pred_softmax)
            acc = tf.reduce_mean(tf.cast(tf.equal(self.Y,tf.cast(tf.argmax(pred,axis=1),tf.int32)),tf.float32))
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
            self.X: (inputs_batch - click_means)/(click_std + 1e-8) ,
        }
        if labels_batch is not None:
            feed_dict[self.Y] = labels_batch
        return feed_dict

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch)
        feed[self.dropout_placeholder] = self.config.dropout
        _, loss = sess.run([self.train_op, self.loss], feed_dict = feed)
        #auc = roc_auc_score(labels_batch,pred1[:,0])
        #print("pred={}".format(pred[:10]))
        return loss

    def evaluate(self, sess, dev_data):
        ce_loss = 0.0
        #auc = 0.0
        acc = 0.0
        total = 0
        predicted = [] #np.zeros((len(dev_data)))
        ground_truth = [] #np.zeros(len(dev_data))
        start = 0
        for (dev_data_x,dev_data_y) in dev_data.get_minibatch(batch_size=800000):
            #dev_data_x, dev_data_y = split_x_y(dev_data)
            feed = self.create_feed_dict(dev_data_x,dev_data_y)
            feed[self.dropout_placeholder] = 1.0
            loss,pred_softmax = sess.run([self.loss,self.pred_softmax], feed_dict=feed)
            closs, aloss,accloss = loss
            total += 1
            if total % 2 ==0:
                print("Evaluate: total={}".format(total))
            ce_loss += closs
            #auc += aloss
            acc += accloss
            predicted.extend(pred_softmax[:,1].tolist())
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
            pred = sess.run(self.pred_softmax,feed_dict= feed)
            preds.extend(pred[:,1])
            labels.extend(y)
        return preds, labels

    def load_best_model(self, sess, saver):
        checkpoint_dir = "newdata/model_{}".format(self.name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Load Model from path:{}".format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

    def fit(self, sess, saver, train_data, dev_data, epochs=None):
        best_auc = 0
        best_step = 0
        step = 0
        if epochs is None:
            epochs = self.config.n_epochs

        print("Total epochs={}".format(epochs))
        for i, (train_x, train_y) in enumerate(train_data.get_minibatch(epochs=epochs)):
            step += 1
            loss,auc,acc = self.train_on_batch(sess, train_x, train_y)

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
                elif i - best_step > 90:
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
            'embed_dim':hp.choice("embed_dim",[20,50]),
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

        model = LogisticRegressionModel(config)
        return model, config

    def train_and_evaluate(self,model,config):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            sess.run(tf.local_variables_initializer())
            auc, step = model.fit(sess,saver,self.train_data,self.dev1_data)

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
