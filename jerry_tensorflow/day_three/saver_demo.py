import tensorflow as tf

# v1 = tf.Variable(tf.constant(1.0), name='v1')
# v2 = tf.Variable(tf.constant(2.0), name='v2')
# result = v1 + v2
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(result))
#     saver.save(sess,'saver/saver_data')
v1 = tf.Variable(tf.constant(3.0), name='v1')
v2 = tf.Variable(tf.constant(4.0), name='v2')
result = v1 + v2

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'saver/saver_data')
    print(sess.run(result))
