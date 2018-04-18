# Tensorflow 支持通过tf.Graph函数来生成新的计算图。不同计算图上的张量和运算都不会共享

import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable('v',initializer=tf.ones_initializer(shape=[1]))

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable('v',initializer=tf.ones_initializer(shape=[1]))
