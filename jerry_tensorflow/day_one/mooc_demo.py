import tensorflow as tf
import numpy as np

# a = [[1, 2, 3]]
# b = [[1], [2], [3]]
# a = np.asarray(a)
# b = np.asarray(b)
# c = a.dot(b)
# print(c)

x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3]))
w2 = tf.Variable(tf.random_normal([3, 1]))
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [1, 1]]})
    print(result)
