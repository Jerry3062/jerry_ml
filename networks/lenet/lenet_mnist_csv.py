import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

train = pd.read_csv('F:/dataset/mnist/train.csv')
test = pd.read_csv('F:/dataset/mnist/test.csv')
test = test.as_matrix()
val = train.iloc[:10000]
train = train.iloc[10000:]

train_y = train.label
train_x = train.drop(['label'], axis=1)
train_y = train_y.as_matrix()
train_y = to_categorical(train_y, 10)
train_x = train_x.as_matrix()

val_y = val.label
val_x = val.drop(['label'], axis=1)
val_y = val_y.as_matrix()
val_y = to_categorical(val_y, 10)
val_x = val_x.as_matrix()

del train
del val

placeholder_x = tf.placeholder(tf.float32, [None, 784])
input_x = tf.reshape(placeholder_x, [-1, 28, 28, 1]) / 255.
placeholder_y = tf.placeholder(tf.float32, [None, 10])

conv1 = tf.layers.conv2d(input_x, 6, [3, 3], 1, 'same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], 2)
conv2 = tf.layers.conv2d(pool1, 16, [3, 3], 1, 'same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, [2, 2], 2)

flat = tf.reshape(pool2, [-1, 7 * 7 * 16])
dense = tf.layers.dense(flat, 120, activation=tf.nn.relu)
dense2 = tf.layers.dense(dense, 84, activation=tf.nn.relu)
logits = tf.layers.dense(dense2, 10)
predictions = tf.argmax(logits, axis=1)
loss = tf.losses.softmax_cross_entropy(onehot_labels=placeholder_y, logits=logits)
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(placeholder_y, axis=1)), tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    for i in range(10000):
        index = i % 100
        batch_x = train_x[index * 320:(index + 1) * 320]
        batch_y = train_y[index * 320:(index + 1) * 320]
        sess.run(train_op, feed_dict={placeholder_x: batch_x, placeholder_y: batch_y})
        sess
        if (i % 100 == 0):
            train_accur = sess.run(accuracy, feed_dict={placeholder_x: batch_x, placeholder_y: batch_y})
            train_acc.append(train_accur)
            train_los = sess.run(loss, feed_dict={placeholder_x: batch_x, placeholder_y: batch_y})
            train_loss.append(train_los)
            val_accur = sess.run(accuracy, feed_dict={placeholder_x: val_x, placeholder_y: val_y})
            val_acc.append(val_accur)
            val_los = sess.run(loss, feed_dict={placeholder_x: val_x, placeholder_y: val_y})
            val_loss.append(val_los)
            print('range%d   train_accur%.6f   train_loss%.6f' % (i, train_accur, train_los))
            print('range%d   val_accur%.6f   val_loss%.6f' % (i, val_accur, val_los))

    # prediction = sess.run(predictions, feed_dict={placeholder_x: test})
    # print(prediction.shape)
    # submission = pd.DataFrame({'ImageId': np.arange(1, 28001), 'Label': prediction})
    # submission.to_csv("submission.csv", index=False)
    sess.close()
    iters = range(len(val_acc))
    plt.figure()
    # acc
    plt.plot(iters, train_acc, 'r', label='train acc')
    # loss
    plt.plot(iters, train_loss, 'g', label='train loss')
    # val_acc
    plt.plot(iters, val_acc, 'b', label='val acc')
    # val_loss
    plt.plot(iters, val_loss, 'k', label='val loss')
    plt.grid(True)
    # plt.xlabel(loss_type)
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.show()
