#coding:utf-8

import tensorflow as tf

from model import *
#from general_utils import Progbar
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import train_test_split

ver="v3"
class Config(object):
    feature_names = ["ip","app","device","os","channel","day","wday","hour","hour_4"] #,"min_6"]
    feature_names_float = [
        "ip_os_dev_count","ip_app_dev_count","ip_app_count","ip_app_wday_hour_count", #4
        "ip_app_day_hour_count","ip_app_hour_count","ip_dev_count", #3
        "qty_wday_hour","qty_day_hour","qty_hour","app_os_count", #4
        "app_count","app_os_dev_count","app_dev_count","ip_app_os_count" #4
    ]
    feature_sizes = [364778,768,4227,956,10,24,6,1021,236,1021,236,1021,236,1021,236,1021,236,1021,236,1021,236,1021,236,1021,236]
    #feature_names = ["app","device","os","channel","day","wday","hour","hour_4"] #,"min_6"]
    #feature_sizes = [768,4227,956,500,10,7,24,6]
    #feature_names = ["day"] #,"min_6"]
    #feature_sizes = [10]
    #feature_names_float = ["app_count"]
    int_features = 25 #len(feature_sizes)
    float_features = 21 #len(feature_names_float)

    feature_num = int_features + float_features
    all_feature_names = feature_names + feature_names_float
    embed_size = 50
    #vocab_size = sum(feature_sizes)
    hidden_size = 512

    lr = 1e-4
    n_steps = -1
    batch_size = 200000
    dropout = 0.5

    #margin threshold
    alpha = 0.5

    # hyparameter for L2 normalization
    lambda1 = 0.01

    n_class = 1 # It should be 1
    n_epochs = 2

feature_names = ["ip","app","device","os","channel","day","wday","hour","hour_4"] #,"min_6"]
feature_names_float = [
        "ip_os_dev_count","ip_app_dev_count","ip_app_count","ip_app_wday_hour_count", #4
        "ip_app_day_hour_count","ip_app_hour_count","ip_dev_count", #3
        "qty_wday_hour","qty_day_hour","qty_hour","app_os_count", #4
        "app_count","app_os_dev_count","app_dev_count","ip_app_os_count" #4
]

def split_x_y(batch,all_feature_names=feature_names + feature_names_float,target="is_attributed"):
    batch.loc[:,'hour_4'] = batch['hour'].apply(lambda x: int(x/4))
    return batch[all_feature_names].values, batch[target].values

def get_minibatch(train_data,batch_size,epochs=None,do_shuffle=True):
    total = len(train_data)
    if do_shuffle:
        train_data = shuffle(train_data)

    #split into positive and negative data
	positive_data = train_data[train_data.is_attributed>0]
	negative_data = train_data[train_data.is_attributed<=0]
	total_positive = len(positive_data)
	total_negative = len(negative_data)
	arr = range(total_negative)

	#get positive batch
	for i in range(100):
		negative_data = shuffle(negative_data)
		arr = shuffle(arr)
		for start in range(0, total_positive, batch_size):
				end = start + batch_size
				if end > total:
					end = total
				batch = train_data[start:end]

				#sample some negative batch
				batch_neg = negative_data[start:end]

                px,py = split_x_y(batch)
                nx,ny = split_x_y(batch_neg)
                yield px,py,nx,ny

def get_minibatch1(train_data,batch_size,epochs= None, do_shuffle=True):
    for (X, y) in train_data.get_minibatch(batch_size=batch_size,epochs= epochs, do_shuffle=do_shuffle):
        total = len(X)

        #split into positive and negative data
        pos_idx = y[y>0]
        neg_idx = y[y<=0]

        pos_X,pos_y = X[pos_idx], y[pos_idx]
        neg_X,neg_y = X[neg_idx], y[neg_idx]
        total_pos = len(pos_X)
        total_neg = len(neg_y)
        arr = range(total_neg)

        #get positive batch
        for i in range(100):
            arr = shuffle(arr)
            nX, ny = neg_X[arr][:total_pos],neg_y[arr][:total_pos]
            for start in range(0, total_pos, batch_size):
				end = start + batch_size
				if end > total_pos:
					end = total_pos
				batch_X, batch_y = pos_X[start:end],pos_y[start:end]

				#sample some negative batch
				batch_neg_X, batch_neg_y = nX[start:end], ny[start:end]

				yield batch_X,batch_y, batch_neg_X,batch_neg_y

"""
Rank Neural net for given one positive and one negative examples
"""
class RankMFNetModel(Model):
    name = "rank_net"

    def __init__(self,config):
        self.config = config
        self.build()

    def add_placeholders(self):
        #ip, app, os, device, os, channel, day, wday, hour, hour/4, min/6
        self.X_pos = tf.placeholder(dtype=tf.float32, shape=[None,self.config.feature_num],name="X_Pos")
        self.X_neg = tf.placeholder(dtype=tf.float32, shape=[None,self.config.feature_num],name="X_Neg")
        self.Y = tf.placeholder(dtype=tf.int32,shape=[None],name="Y")
        self.dropout_placeholder = tf.placeholder(tf.float32)

        # Initialize the embeddings parameters
        with tf.variable_scope("embed") as scope:
            self.embeddings = []
            self.embeddings1 = []   # for one dimension embeddings
            for (idx, size) in enumerate(self.config.feature_sizes):
                embeddings = tf.Variable(
					tf.random_uniform([size, self.config.embed_size], -1.0, 1.0)*0.001,
					name="embeddings_{}".format(idx)
				)
                self.embeddings.append(embeddings)

                embeddings1 = tf.Variable(
                    tf.random_uniform([size, 1], -1.0, 1.0)*0.001,
                    name="embeddings1_{}".format(idx)
                )
                self.embeddings1.append(embeddings1)

        # Initialize the predict parameters
        with tf.variable_scope("predict") as scope:
            # Weights for one dimension embeddings
            self.W0 = tf.Variable(
                tf.random_uniform([self.config.int_features,1])*0.001,
                name = "W0")

            # Weights for dimension inner production
            self.W1 = tf.Variable(
                tf.random_uniform([self.config.int_features,self.config.int_features])*0.001,
                name = "W1")

            # Global bias
            self.b = tf.zeros(shape=1, name = "b")

            # Weights for float features
            self.W2 = tf.Variable(
                tf.random_uniform([self.config.float_features,1])*0.001,
                name = "W2")

    def add_embedding(self,X):
        with tf.variable_scope("embed") as scope:
            embeds = []
            embeds1 = []
            for (idx, size) in enumerate(self.config.feature_sizes):
                embeddings = self.embeddings[idx]
                embed = tf.nn.embedding_lookup(embeddings,tf.slice(X,[0,idx],[-1,1]))
                embeds.append(embed)

                embeddings1 = self.embeddings1[idx]
                embed1 = tf.nn.embedding_lookup(embeddings1,tf.slice(X,[0,idx],[-1,1]))
                embeds1.append(embed1)

            embeds = tf.concat(embeds,axis=1)
            embeds1 = tf.concat(embeds1,axis=1)
        return embeds, embeds1

    def add_prediction_op(self):
        r1 = self.predict_example(self.X_pos)
        r2 = self.predict_example(self.X_neg)
        return r1 - r2

    def predict_example(self,Input_X):
        X1 = tf.slice(Input_X,begin=[0,0],size=[-1,self.config.int_features])
        X2 = tf.slice(Input_X,begin=[0,self.config.int_features],size=[-1,self.config.float_features])
        #print(tf.shape(X1))
        #print(tf.shape(X2))
        X1 = tf.cast(X1, tf.int32)
        X_First,X_First1 = self.add_embedding(X1)

        X_Second = tf.cast(X2,tf.float32)
        with tf.variable_scope("predict") as scope:
            # sum of one dimension embedding
            pred1 = tf.reduce_sum(X_First1 * self.W0, axis=[1,2])  #(None, int_features, 1)

            # sum of multiple dimension embeddings
            # X_First (None, int_feature, embed_size)
            X_First_transp = tf.transpose(X_First,[0,2,1])
            # (None, int_features, int_features)
            products = tf.matmul(X_First, X_First_transp)
            pred2 = tf.reduce_sum(products * self.W1, axis=[1,2])

            pred = pred1 + pred2 + self.b

            pred += tf.matmul(X2,self.W2)

            return pred

    def add_loss_op(self, pred):
        with tf.variable_scope("loss") as scope:
            #loss = tf.maximum((self.config.alpha - pred, 0))
            loss = tf.reduce_mean(-tf.log(tf.nn.sigmoid(pred)))
            if self.config.lambda1 >0.0:
                l2_loss = self.config.lambda1 * (
                    tf.reduce_sum(self.W1 * self.W1) + tf.reduce_sum(self.W2*self.W2) + tf.reduce_sum(self.W0* self.W0)
                )
                loss += l2_loss
            return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def create_feed_dict(self, pos_batch, t1, neg_batch,t2):
        feed_dict = {
            self.X_pos: pos_batch,
            self.X_neg: neg_batch,
        }
        return feed_dict

    def train_on_batch(self, sess, px_batch, nx_batch):
        feed = self.create_feed_dict(px_batch, None, nx_batch, None)
        feed[self.dropout_placeholder] = self.config.dropout
        _, loss = sess.run([self.train_op, self.loss], feed_dict = feed)
        return loss

    def evaluate(self, sess, dev_data):
        total_loss = 0.0
        total = 0
        for (px,py,nx,ny) in get_minibatch(dev_data,self.config.batch_size,do_shuffle=False,epochs=1) :
            #dev_data_x, dev_data_y = split_x_y(dev_data)
            feed = self.create_feed_dict(px,py,nx,ny)
            feed[self.dropout_placeholder] = 1.0
            loss = sess.run(self.loss, feed_dict=feed)
            total += 1
            #if total % 10 ==0:
            #    print("Evaluate: total={}".format(total))
            total_loss += loss
        return total_loss/float(total)

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
        raise "Not implementated!!"
        X, y = split_x_y(inputs_batch)
        feed = self.create_feed_dict(X)
        feed[self.dropout_placeholder] = 1.0
        pred = sess.run(self.pred_softmax,feed_dict= feed)
        return pred[:,1]

    def run_epoch(self, sess, epoch,train_data, test_data):
        #prog = Progbar(target=1 + len(train_data)/ self.config.batch_size)
        for i, (px, py, nx, ny) in enumerate(get_minibatch(train_data, self.config.batch_size)):
            loss = self.train_on_batch(sess, px, nx)
            #prog.update(i+1, [("train_loss",loss),("train_auc",auc)])
            if i % 10 == 0:
                print("Epoch:{}, step:{}, loss:{}".format(epoch, i, loss))

            if i % 50 == 0:
                dev_loss = self.evaluate(sess,test_data)
                #prog.update(i+1, [("dev_loss",dev_loss),("dev_auc",dev_auc)])
                print("==>Evaluate: epoch:{}, step:{}, loss:{}".format(epoch, i, dev_loss))

        dev_loss = self.evaluate(sess, test_data)
        return dev_loss
        #return loss, auc

    def fit(self, sess, saver, train_data, dev_data):
        best_loss = 0
        for i, (px, py, nx, ny) in enumerate(get_minibatch(train_data, self.config.batch_size,epochs=self.config.n_epochs)):
            loss = self.train_on_batch(sess, px, nx)
            #prog.update(i+1, [("train_loss",loss),("train_auc",auc)])
            if i % 10 == 0:
                print("Step:{}, loss:{}".format(i, loss))

            if i % 50 == 0:
                dev_loss = self.evaluate(sess,dev_data)
                print("==>Evaluate: Step:{}, loss:{}".format(i, dev_loss))
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    best_step = i
                    print("***Best model: step:{}, loss:{}".format(i, dev_loss))
                    save_path = saver.save(sess,"./data/model_{}/{}".format(self.name,self.name))
                    print("Save in path = {}".format(save_path))
                elif i - best_step > 90:
                    return dev_loss, best_step

            if self.config.n_steps >0 and i >= self.config.n_steps:
                print("***Max step reaches:step={},max={}".format(i,self.config.n_steps))
                break

        dev_loss = self.evaluate(sess,dev_data)
        print("==>End Evaluate: loss:{}".format(dev_loss))
        if dev_loss < best_loss:
            best_auc = dev_loss
            print("***Best model: loss:{}".format(dev_loss))
            save_path = saver.save(sess,"./data/model_{}/{}".format(self.name,self.name))
            print("Save in path = {}".format(save_path))

        return best_loss, best_step

import time
import pandas as pd
import sys

def main(debug=True):
    print("Load train data...")
    #train_data = pd.read_hdf("data/train_s.hdf","data")
    #train_data = pd.read_hdf("data/train.hdf","data")
    #train_df = pd.read_hdf("data/train.hdf","data",start= 131886954)
    #train_df = pd.read_hdf("data/train.hdf","data",start= 131886954)
    #train_df = pd.read_hdf("data/train.hdf","data")
    """
    all_train_df = train_df[
        ((train_df.hour==4)|(train_df.hour==5)| \
        (train_df.hour==9)|(train_df.hour==10) \
        |(train_df.hour==13)|(train_df.hour==14))
    ]
	"""

    train_df = pd.read_hdf("data/train_s.hdf","data")
    all_train_df = train_df
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

    config = Config()
    model = RankMFNetModel(config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    do_gen = False
    with tf.Session() as session:
        session.run(init)
        session.run(tf.local_variables_initializer())

        #model.fit(session, saver, train_examples, dev_examples)
        #Use all train data to generate the submissions
        if do_gen:
            model.fit(session, saver, all_train_df, dev_examples)
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
            file ='sub_LR_emd_{}_hidden_{}_lr_{}_epoch_{}.csv'.format(
                model.config.embed_size,
                model.config.hidden_size,
                model.config.lr,
                model.config.n_epochs,
            )
            print("writing {}......".format(file))
            sub.to_csv(file,index=False)

if __name__ == "__main__":
    main()
