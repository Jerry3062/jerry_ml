# Tensorflow 支持通过tf.Graph函数来生成新的计算图。不同计算图上的张量和运算都不会共享

# import tensorflow as tf
# g1 = tf.Graph()
# with g1.as_default():
#     v = tf.get_variable('v',initializer=tf.ones_initializer(shape=[1]))
#
# g2 = tf.Graph()
# with g2.as_default():
#     v = tf.get_variable('v',initializer=tf.ones_initializer(shape=[1]))

# import random
# import numpy as np
#
# a = [1, 2, 3]
# a = np.asarray(a)
# random.shuffle(a)
# print(a)

# import cv2
# a = cv2.imread('scikit-learn.png',1)
# cv2.imshow("a",a)
# cv2.waitKey(0)

import tensorflow as tf

one_hot_encoded = tf.constant([[0, 0, 0, 1],
                               [1, 0, 0, 0]])
decoded = tf.argmax(one_hot_encoded, axis=1)
print(tf.Session().run(decoded))
