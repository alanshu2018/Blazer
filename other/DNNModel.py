#coding:utf-8

import tensorflow as tf

from model import *
from general_utils import Progbar
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import numpy as np

class Config(object):
    feature_names = ["ip","app","device","os","channel","day","wday","hour","hour_4"] #,"min_6"]
    feature_sizes = [364778,768,4227,956,500,10,7,24,6]
    feature_sizes_add =[0] + feature_sizes[:-1]
    int_features = len(feature_sizes)
    float_features = 15
    embed_size = 100
    vocab_size = sum(feature_sizes)
    hidden_size = 1000

    n_class = 1
    lr = 1e-2
    n_epochs = 10
    batch_size = 20000
    dropout = 1.0

def split_x_y(batch):
    batch['hour_4'] = batch['hour'].apply(lambda x: int(x/4))
    return batch[["ip","app","device","os","channel","day","wday","hour","hour_4",
                  "ip_os_dev_count","ip_app_dev_count","ip_app_count","ip_app_wday_hour_count", #4
                  "ip_app_day_hour_count","ip_app_hour_count","ip_dev_count", #3
                  "qty_wday_hour","qty_day_hour","qty_hour","app_os_count", #4
                  "app_count","app_os_dev_count","app_dev_count","ip_app_os_count" #4
                  ]].values,\
           batch["is_attributed"].values

def get_minibatch(train_data,batch_size,do_shuffle=True):
    total = len(train_data)
    if do_shuffle:
        train_data = shuffle(train_data)

    for start,end in zip(range(0, total, batch_size),range(batch_size,total,batch_size)):
        batch = train_data[start:end]
        yield split_x_y(batch)


class DNNModel(Model):
    name = "dnn"
    config = Config()

    def __init__(self):
        self.build()

    def add_placeholders(self):
        #ip, app, os, device, os, channel, day, wday, hour, hour/4, min/6
        self.X1 = tf.placeholder(dtype=tf.int32, shape=[None,self.config.int_features],name="X1")
        self.X2 = tf.placeholder(dtype=tf.float32,shape=[None,self.config.float_features],name="X2")
        self.Y = tf.placeholder(dtype=tf.int32,shape=[None],name="Y")
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def add_embedding(self):
        with tf.variable_scope("embed") as scope:
            self.word_embedding =tf.Variable(
                tf.random_uniform([self.config.vocab_size,self.config.embed_size],-1.0,1.0),
                name="embeddings")
            embeddings = tf.nn.embedding_lookup(self.word_embedding,self.X1)
            embeddings = tf.reshape(embeddings,[-1, self.config.int_features * self.config.embed_size])

        return embeddings

    def add_prediction_op(self):
        x1 = self.add_embedding()
        x = tf.concat([x1,self.X2],axis=1)
        with tf.variable_scope("predict") as scope:
            W1 = tf.Variable(
                tf.random_uniform([self.config.int_features * self.config.embed_size + self.config.float_features,
                                     self.config.hidden_size])*0.01,
                name = "W1")
            b1 = tf.zeros(shape=self.config.hidden_size, name = "b1")
            W2 = tf.Variable(
                tf.random_uniform([self.config.hidden_size,
                                     self.config.n_class])*0.01,
                name = "W2")
            b2 = tf.zeros(shape=(self.config.n_class),name="b2")

            # swish
            h = tf.matmul(x,W1) + b1
            h = tf.nn.sigmoid(h) * h
            h_drop = tf.nn.dropout(h, self.dropout_placeholder)

            pred = tf.matmul(h_drop, W2) + b2
            #pred = tf.nn.softmax(pred)
            self.pred1 = tf.nn.sigmoid(pred)

            return pred

    def add_loss_op(self, pred):
        with tf.variable_scope("loss") as scope:
            shape = tf.shape(pred)
            Y = tf.reshape(tf.cast(self.Y,tf.float32),[shape[0]])
            prob = tf.reshape(tf.nn.sigmoid(pred),[shape[0]])
            #loss = - Y * tf.log(prob) - (1-Y) * tf.log(1-prob)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=tf.reshape(pred,[shape[0]]))
            loss = tf.reduce_mean(loss)
            #p = tf.nn.softmax(pred)
            _, auc = tf.metrics.auc(Y,prob)
            #acc = tf.reduce_mean(tf.cast(tf.equal(self.Y,tf.cast(tf.argmax(pred,axis=1),tf.int32)),tf.float32))
            return loss, auc

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        l,a = loss
        train_op = optimizer.minimize(l)
        return train_op

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        inputs_batch[:,:self.config.int_features] += self.config.feature_sizes_add
        feed_dict = {
            self.X1: inputs_batch[:,:self.config.int_features],
            self.X2:inputs_batch[:,self.config.int_features:]
        }
        if labels_batch is not None:
            feed_dict[self.Y] = labels_batch
        return feed_dict

    def train_on_batch(self, sess, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch)
        feed[self.dropout_placeholder] = self.config.dropout
        _, loss,pred1 = sess.run([self.train_op, self.loss, self.pred1], feed_dict = feed)
        #auc = roc_auc_score(labels_batch,pred1[:,0])
        #print("pred={}".format(pred[:10]))
        return loss

    def evaluate(self, sess, dev_data):
        dev_data_x, dev_data_y = split_x_y(dev_data)
        feed = self.create_feed_dict(dev_data_x,dev_data_y)
        feed[self.dropout_placeholder] = 1.0
        loss,pred = sess.run([self.loss,self.pred1], feed_dict=feed)
        #auc = roc_auc_score(dev_data_y,pred[:,0])
        #print("auc={}".format(auc))
        return loss #[0],auc

    def run_epoch(self, sess, epoch,train_data, test_data):
        #prog = Progbar(target=1 + len(train_data)/ self.config.batch_size)
        for i, (train_x, train_y) in enumerate(get_minibatch(train_data, self.config.batch_size)):
            loss,auc = self.train_on_batch(sess, train_x, train_y)
            #prog.update(i+1, [("train_loss",loss),("train_auc",auc)])
            if i % 100 == 0:
                print("Epoch:{}, step:{}, train_loss:{},train_auc:{}".format(epoch, i, loss, auc))

            if i % 1000 == 0:
                dev_loss,dev_auc = self.evaluate(sess,train_data)
                #prog.update(i+1, [("dev_loss",dev_loss),("dev_auc",dev_auc)])
                print("Evaluate: epoch:{}, step:{}, dev_loss:{},dev_auc:{}".format(epoch, i, dev_loss, dev_auc))

        dev_loss = self.evaluate(sess, test_data)
        return dev_loss

    def fit(self, sess, saver, train_data, dev_data):
        best_auc = 0
        for epoch in range(self.config.n_epochs):
            dev_loss,dev_auc = self.run_epoch(sess, epoch, train_data, dev_data)
            print("dev_auc={}".format(dev_auc))
            print("best_auc={}".format(best_auc))
            if dev_auc > best_auc:
                best_auc = dev_auc
                print("Best model: epoch={}, loss={}, auc={}".format(epoch, dev_loss, dev_auc))
                if saver:
                    saver.save(sess,"./data/{}_model".format(self.name))

import time
import pandas as pd
import tables
import sys

def main(debug=True):
    print("Load data...")
    train_data = pd.read_hdf("data/train_s.hdf","data")
    #train_data = pd.read_hdf("data/train.hdf","data")
    total = len(train_data)
    train_cnt = int(total * 0.98)
    train_data = shuffle(train_data)
    """
    train_data[:100000].to_hdf("data/train_s.hdf","data")
    sys.exit(-1)
    """
    train_examples = train_data[:train_cnt]
    dev_examples = train_data[train_cnt:]

    print("train_cnt={}, dev_cnt={}".format(train_cnt,total-train_cnt))

    print("Building model...")
    config = Config()
    start = time.time()

    model = DNNModel()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        session.run(tf.local_variables_initializer())

        model.fit(session, saver, train_examples, dev_examples)
    end = time.time()
    print("Cost {} seconds".format(end-start))


if __name__ == "__main__":
    main()
