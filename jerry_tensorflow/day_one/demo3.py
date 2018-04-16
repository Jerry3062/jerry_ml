import tensorflow as tf

import numpy as np

print(np.__version__)

matrix1 = tf.constant([[2.0, 2.0]])

matrix2 = tf.constant([[2.0], [2.0]])

product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
result = sess.run(product)
print(type(result))

sess.close()
