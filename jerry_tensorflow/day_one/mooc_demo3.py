import tensorflow as tf

w = tf.Variable(tf.constant(4,dtype=tf.float32))
loss = tf.square(w + 1)
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        sess.run(train_step)
    print(sess.run(loss), sess.run(w))
    tf.summary.FileWriter("F:/graph", sess.graph)
