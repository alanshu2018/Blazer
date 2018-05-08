#coding:utf-8

import tensorflow as tf

from model import *
#from general_utils import Progbar
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split

class Config(object):
    feature_names = ["ip","app","device","os","channel","day","wday","hour","hour_4"] #,"min_6"]
    feature_names_float = [
        "ip_os_dev_count","ip_app_dev_count","ip_app_count","ip_app_wday_hour_count", #4
        "ip_app_day_hour_count","ip_app_hour_count","ip_dev_count", #3
        "qty_wday_hour","qty_day_hour","qty_hour","app_os_count", #4
        "app_count","app_os_dev_count","app_dev_count","ip_app_os_count" #4
    ]
    feature_sizes = [364778,768,4227,956,500,10,7,24,6]
    #feature_names = ["app","device","os","channel","day","wday","hour","hour_4"] #,"min_6"]
    #feature_sizes = [768,4227,956,500,10,7,24,6]
    #feature_names = ["day"] #,"min_6"]
    #feature_sizes = [10]
    #feature_names_float = ["app_count"]

    int_features = len(feature_sizes)
    float_features = len(feature_names_float)

    # Total feature number
    feature_num = int_features + float_features
    all_feature_names = feature_names + feature_names_float
    embed_size = 50
    #vocab_size = sum(feature_sizes)
    hidden_size = 512

    #margin threshold
    alpha = 0.5

    # hyparameter for L2 normalization
    lambda1 = 0 #0.01

    n_class = 1 # It should be 1
    lr = 1e-5
    n_epochs = 40
    batch_size = 200000
    dropout = 0.5

g_config = Config()

def split_x_y(batch,all_feature_names=g_config.all_feature_names,target="is_attributed"):
    batch.loc[:,'hour_4'] = batch['hour'].apply(lambda x: int(x/4))
    return batch[all_feature_names].values, batch[target].values

def get_minibatch1(train_data,batch_size,do_shuffle=True,loops=50):
    total = len(train_data)
    if do_shuffle:
        train_data = shuffle(train_data)

        #split into positive and negative data
        if True:
            for start in range(0, total, batch_size):
                end = start + batch_size
                if end > total:
                    end = total
                batch = train_data[start:end]

                yield split_x_y(batch)

def get_minibatch(train_data,batch_size,do_shuffle=True,loops=200):
    total = len(train_data)
    if do_shuffle:
        train_data = shuffle(train_data)

        #split into positive and negative data
        positive_data = train_data[train_data.is_attributed>0]
        negative_data = train_data[train_data.is_attributed<=0]
        total_positive = len(positive_data)
        total_negative = len(negative_data)

        #get positive batch
        for i in range(loops):
            negative_data = shuffle(negative_data)
            for start in range(0, total_positive, batch_size):
                end = start + batch_size
                if end > total_positive:
                    end = total_positive
                if end > total_negative:
                    end = total_negative
                batch = positive_data[start:end]

                #sample some negative batch
                batch_neg = negative_data[start:end]
                #print("batch.positive.size={}".format(len(batch)))
                #print("batch.negative.size={}".format(len(batch_neg)))

                yield split_x_y(batch), split_x_y(batch_neg)

"""
Rank Neural net for given one positive and one negative examples
"""
class RankNetModel(Model):
    name = "rank_net"

    def __init__(self,config):
        self.config = config
        self.build()

    def add_placeholders(self):
        #ip, app, os, device, os, channel, day, wday, hour, hour/4, min/6
        self.X_pos = tf.placeholder(dtype=tf.int32, shape=[None,self.config.feature_num],name="X_Pos")
        self.X_neg = tf.placeholder(dtype=tf.int32, shape=[None,self.config.feature_num],name="X_Neg")
        self.Y = tf.placeholder(dtype=tf.int32,shape=[None],name="Y")
        self.dropout_placeholder = tf.placeholder(tf.float32)

        # Initialize the embeddings parameters
        with tf.variable_scope("embed") as scope:
            self.embeddings = []
            for (idx, size) in enumerate(self.config.feature_sizes):
                embeddings = tf.Variable(
                    tf.random_uniform([size, self.config.embed_size], -1.0, 1.0)*0.01,
                    name="embeddings_{}".format(idx)
                )
                self.embeddings.append(embeddings);

            # Initialize the predict parameters
        with tf.variable_scope("predict") as scope:
            self.W1 = tf.Variable(
                tf.random_uniform([self.config.int_features*self.config.embed_size + self.config.float_features,
                                   self.config.hidden_size])*0.0001,
                name = "W1")
            self.b1 = tf.zeros(shape=self.config.hidden_size, name = "b1")
            self.W2 = tf.Variable(
                tf.random_uniform([self.config.hidden_size,
                                   self.config.n_class])*0.0001,
                name = "W2")
            self.b2 = tf.zeros(shape=(self.config.n_class),name="b2")

    def add_embedding(self,X):
        with tf.variable_scope("embed") as scope:
            embeds = []
            for (idx, size) in enumerate(self.config.feature_sizes):
                embeddings = self.embeddings[idx]
                embed = tf.nn.embedding_lookup(embeddings,tf.slice(X,[0,idx],[-1,1]))
                embeds.append(embed)
            embeds = tf.concat(embeds,axis=1)
        return embeds

    def add_prediction_op(self):
        self.r1 = self.predict_example(self.X_pos)
        self.r2 = self.predict_example(self.X_neg)
        return self.r1 - self.r2

    def predict_example(self,Input_X):
        X1 = tf.slice(Input_X,begin=[0,0],size=[-1,self.config.int_features])
        X2 = tf.slice(Input_X,begin=[0,self.config.int_features],size=[-1,self.config.float_features])
        #print(tf.shape(X1))
        #print(tf.shape(X2))
        X_First = self.add_embedding(X1)
        X_First = tf.reshape(X_First,[-1, self.config.int_features*self.config.embed_size])
        X_Second = tf.layers.batch_normalization(tf.cast(X2,tf.float32))
        X = tf.concat([X_First,X_Second],axis=-1)
        #X_shape = tf.shape(X)
        #X = X_First

        with tf.variable_scope("predict") as scope:
            # swish
            h = tf.matmul(X,self.W1) + self.b1
            h = tf.nn.sigmoid(h) * h
            h_drop = tf.nn.dropout(h, self.dropout_placeholder)

            pred = tf.matmul(h_drop, self.W2) + self.b2
            pred = tf.nn.sigmoid(pred)
            #self.pred_softmax = tf.nn.softmax(pred)

            return pred

    def add_loss_op(self, pred):
        with tf.variable_scope("loss") as scope:
            shape = tf.shape(self.Y)
            r1 = tf.reshape(self.r1,shape)
            Y = tf.cast(self.Y,tf.float32)
            self.loss_single = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=r1))
            _, self.auc_single = tf.metrics.auc(Y,r1)
            self.acc_single = tf.reduce_mean(tf.cast(tf.equal(self.Y,tf.cast(r1,tf.int32)),tf.float32))

            #Hinge loss
            loss = tf.reduce_mean(tf.maximum(self.config.alpha - pred, 0))
            #
            #loss = -pred + tf.log(1 + tf.exp(pred)) #tf.reduce_mean(tf.maximum(self.config.alpha - pred, 0))
            #loss = tf.reduce_mean(loss)
            if self.config.lambda1 >0.0:
                l2_loss = self.config.lambda1 * (tf.reduce_sum(self.W1 * self.W1) + tf.reduce_sum(self.W2*self.W2))
                loss += l2_loss
            return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def create_feed_dict_single(self, X, y):
        feed_dict = {
            self.X_pos: X,
            self.Y: y,
        }
        return feed_dict

    def create_feed_dict(self, pos_batch, neg_batch):
        feed_dict = {
            self.X_pos: pos_batch,
            self.X_neg: neg_batch,
        }
        return feed_dict

    def train_on_batch(self, sess, px_batch, nx_batch):
        feed = self.create_feed_dict(px_batch, nx_batch)
        feed[self.dropout_placeholder] = self.config.dropout
        _, loss = sess.run([self.train_op, self.loss], feed_dict = feed)
        return loss

    def evaluate1(self, sess, dev_data):
        total_loss = 0.0
        total = 0
        for ((px,py),(nx,ny)) in get_minibatch(dev_data,self.config.batch_size,loops=1) :
            #dev_data_x, dev_data_y = split_x_y(dev_data)
            feed = self.create_feed_dict(px,nx)
            feed[self.dropout_placeholder] = 1.0
            loss = sess.run(self.loss, feed_dict=feed)
            total += 1
            #if total % 10 ==0:
            #    print("Evaluate: total={}".format(total))
            total_loss += loss
        return total_loss/float(total)

    def evaluate(self, sess, dev_data):
        total = 0
        total_acc = 0
        total_auc = 0
        total_loss = 0
        predicted = np.zeros((len(dev_data)))
        ground_truth = np.zeros(len(dev_data))
        start = 0
        for ((px,py)) in get_minibatch1(dev_data,self.config.batch_size,loops=1) :
            #dev_data_x, dev_data_y = split_x_y(dev_data)
            feed = self.create_feed_dict_single(px,py)
            feed[self.dropout_placeholder] = 1.0
            loss,acc,auc,r1 = sess.run([self.loss_single,self.acc_single,self.auc_single,self.r1], feed_dict=feed)
            total += 1
            predicted[start:start + len(px)] = np.reshape(r1,(r1.shape[0]))
            ground_truth[start:start + len(px)] = py
            total_loss += loss
            total_acc += acc
            total_auc += auc

        auc = roc_auc_score(ground_truth, predicted)
        return total_loss/float(total), total_acc/float(total), auc #total_auc/float(total)

    def predict(self, sess, inputs_batch):
        raise "Not implementated!!"
        preds = []
        for X, y in get_minibatch(inputs_batch,self.config.batch_size,do_shuffle=False):
            feed = self.create_feed_dict(X)
            feed[self.dropout_placeholder] = 1.0
            pred = sess.run(self.pred_softmax,feed_dict= feed)
            preds.extend(pred[:,1])
        return preds

    def predict_on_batch(self, sess, inputs_batch):
        X, y = split_x_y(inputs_batch)
        feed = self.create_feed_dict(X)
        feed[self.dropout_placeholder] = 1.0
        pred = sess.run(self.pred_softmax,feed_dict= feed)
        return pred[:,1]

    def run_epoch(self, sess, epoch,train_data, test_data):
        #prog = Progbar(target=1 + len(train_data)/ self.config.batch_size)
        for i, ((px, py), (nx, ny)) in enumerate(get_minibatch(train_data, self.config.batch_size)):
            loss = self.train_on_batch(sess, px, nx)
            #prog.update(i+1, [("train_loss",loss),("train_auc",auc)])
            if i % 10 == 0:
                print("Epoch:{}, step:{}, loss:{}".format(epoch, i, loss))

            if i % 50 == 0:
                dev_loss,dev_acc,dev_auc = self.evaluate(sess,test_data)
                #prog.update(i+1, [("dev_loss",dev_loss),("dev_auc",dev_auc)])
                print("==>Evaluate: epoch:{}, step:{}, loss:{}, acc:{}, auc:{}".format(epoch, i, dev_loss,dev_acc,dev_auc))

        dev_loss,dev_acc,dev_auc = self.evaluate(sess, test_data)
        return dev_loss, dev_acc,dev_auc
        #return loss, auc

    def fit(self, sess, saver, train_data, dev_data):
        best_loss = 0
        for epoch in range(self.config.n_epochs):
            dev_loss,dev_acc,dev_auc = self.run_epoch(sess, epoch, train_data, dev_data)
            print("dev_loss={},dev_acc={},dev_auc={}".format(dev_loss,dev_acc,dev_auc))
            if dev_loss < best_loss:
                best_loss = dev_loss
                print("Best model: epoch:{}, loss:{},acc:{},auc:{}".format(epoch, dev_loss,dev_acc,dev_auc))
                if saver:
                    saver.save(sess,"./data/{}_model".format(self.name))

import time
import pandas as pd
import sys

def main(debug=True):
    print("Load train data...")
    #train_data = pd.read_hdf("data/train_s.hdf","data")
    #train_data = pd.read_hdf("data/train.hdf","data")
    train_df = pd.read_hdf("data/train.hdf","data",start= 131886954)
    #train_df = pd.read_hdf("data/train.hdf","data",start= 161886954)
    #train_df = pd.read_hdf("data/train.hdf","data")

    """
	all_train_df = train_df[
        ((train_df.hour==4)|(train_df.hour==5)| \
        (train_df.hour==9)|(train_df.hour==10) \
        |(train_df.hour==13)|(train_df.hour==14))
    ]
    """
    all_train_df = train_df
    #train_df = pd.read_hdf("data/train_s.hdf","data")
    #all_train_df = train_df
    #use positive example and 1/10 of negative example
    #len_train = len(all_train_df)
    train_examples, dev_examples = train_test_split(all_train_df,test_size=0.1,shuffle=True)
    #train_examples = all_train_df[:(len_train-3000000)]
    #dev_examples = all_train_df[(len_train-3000000):len_train]

    #train_data = shuffle(train_data)
    """
    train_data[:100000].to_hdf("data/train_s.hdf","data")
    sys.exit(-1)
    """
    #train_examples = train_data[:train_cnt]
    #dev_examples = train_data[train_cnt:]

    print("train_cnt={}, dev_cnt={}".format(len(train_examples),len(dev_examples)))

    print("Building model...")
    start = time.time()

    model = RankNetModel(g_config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    do_gen = True
    with tf.Session() as session:
        session.run(init)
        session.run(tf.local_variables_initializer())

        #model.fit(session, saver, train_examples, dev_examples)
        #Use all train data to generate the submissions
        if do_gen:
            #model.fit(session, saver, all_train_df, dev_examples)
            model.fit(session, saver, train_examples, dev_examples)
        else:
            model.fit(session, saver, train_examples, dev_examples)

        end = time.time()
        print("Training Cost {} seconds".format(end-start))

        if do_gen:
            # predict
            print ("Load test data")
            test_data = pd.read_hdf("data/test.hdf","data")
            #print(test_data.head())
            sub = pd.DataFrame()
            sub['click_id'] = test_data['click_id'].astype('int')
            print("Predicting...")
            sub['is_attributed'] = model.predict(session,test_data)
            file ='sub_RankNet_emd_{}_hidden_{}_lr_{}_epoch_{}.csv'.format(
                model.config.embed_size,
                model.config.hidden_size,
                model.config.lr,
                model.config.n_epochs,
            )
            print("writing {}......".format(file))
            sub.to_csv(file,index=False)

if __name__ == "__main__":
    main()
