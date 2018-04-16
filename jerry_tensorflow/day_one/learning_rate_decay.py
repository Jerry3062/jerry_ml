import tensorflow as tf

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_STEP = 1

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    LEARNING_RATE_STEP,
    LEARNING_RATE_DECAY,
    staircase=True
)

w = tf.Variable(tf.constant(5, dtype=tf.float32))
loss = tf.square(w + 1)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss
                                                                       , global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        sess.run(train_step)
        learning_rate1 = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("train_step", train_step)
        print("learning_rate", learning_rate1)
        print("global_step_val", global_step_val)
        print("w_val", w_val)
        print("loss_val", loss_val)
