import tensorflow as tf
import numpy as np

batch_size,seq_len = 3, 10
position_size = 6.
position_j1 = 2 * tf.range(position_size/2)
position_j = 1. / tf.pow(1000., 2 * tf.range(position_size/2,dtype=tf.float32)/position_size)
position_j = tf.expand_dims(position_j,0)
position_i = tf.range(tf.cast(seq_len,tf.float32), dtype=tf.float32)
position_i = tf.expand_dims(position_i, 1)
position_ij = tf.matmul(position_i,position_j)
position_ij = tf.concat([tf.cos(position_ij),tf.sin(position_ij)],1)
position_embedding = tf.expand_dims(position_ij,0) + tf.zeros((batch_size,seq_len, position_size))

tf.nn.conv1d()
def print_vector(name,p):
    print("Vector {}, shape={}".format(name,p.shape))
    print("Vector={}".format(p))

with tf.Session() as sess:
    pj1,pj,pi,pij,pe = sess.run([position_j1,position_j,position_i,position_ij,position_embedding])
    print_vector("pj1",pj1)
    print_vector("pj",pj)
    print_vector("pi",pi)
    print_vector("pij",pij)
    print_vector("pe",pe)

