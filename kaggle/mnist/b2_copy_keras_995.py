import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import time

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

conv1 = tf.layers.conv2d(input_x, 64, [3, 3], 1, 'same')
conv1 = tf.layers.batch_normalization(conv1)
conv1 = tf.nn.relu(conv1)
conv1 = tf.layers.batch_normalization(conv1)

conv2 = tf.layers.conv2d(conv1, 64, [3, 3], 1, 'same')
conv2 = tf.layers.batch_normalization(conv2)
conv2 = tf.nn.relu(conv2)
conv2 = tf.layers.batch_normalization(conv2)

poo1 = tf.layers.max_pooling2d(conv2, [2, 2], 2)

flat = tf.reshape(poo1, [-1, 14 * 14 * 64])

dense1 = tf.layers.dense(flat, 64)
dense1 = tf.layers.batch_normalization(dense1)
dense1 = tf.nn.relu(dense1)

dense2 = tf.layers.dense(dense1, 64)
dense2 = tf.layers.batch_normalization(dense2)
dense2 = tf.nn.relu(dense2)

dense3 = tf.layers.dense(dense2, 10)

logits = tf.layers.batch_normalization(dense3)
predictions = tf.argmax(logits, axis=1)
loss = tf.losses.softmax_cross_entropy(onehot_labels=placeholder_y, logits=logits)
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1), tf.argmax(placeholder_y, axis=1)), tf.float32))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    start_time = time.time()
    for i in range(10001):
        index = i % 100
        batch_x = train_x[index * 320:(index + 1) * 320]
        batch_y = train_y[index * 320:(index + 1) * 320]
        sess.run(train_op, feed_dict={placeholder_x: batch_x, placeholder_y: batch_y})
        if (i % 100 == 0):
            train_accur = sess.run(accuracy, feed_dict={placeholder_x: batch_x, placeholder_y: batch_y})
            train_acc.append(train_accur)
            train_los = sess.run(loss, feed_dict={placeholder_x: batch_x, placeholder_y: batch_y})
            train_loss.append(train_los)
            # logit_arr = []
            # for j in range(100):
            #     val_x_batch = val_x[j * 500:(j + 1) * 500]
            #     logit_item = sess.run(logits, feed_dict={placeholder_x: val_x_batch})
            #     logit_arr.append(logit_item)
            # val_accur = np.argmax(logit_arr)
            # val_acc.append(val_accur)
            # val_los = sess.run(loss, feed_dict={placeholder_x: val_x, placeholder_y: val_y})
            # val_loss.append(val_los)
            print('range%d   train_accur%.6f   train_loss%.6f' % (i, train_accur, train_los))
            # print('range%d   val_accur%.6f   val_loss%.6f\n' % (i, val_accur, val_los))

        # prediction = sess.run(predictions, feed_dict={placeholder_x: test})
        # print(prediction.shape)
        # submission = pd.DataFrame({'ImageId': np.arange(1, 28001), 'Label': prediction})
        # submission.to_csv("submission.csv", index=False)
    file = open('save_info_from_bn_cnn', 'a')
    # file.write('local response normalizatidon lenet,cost %.3fseconds \n' % (time.time() - start_time))
    file.write('cost %.3fseconds \n' % (time.time() - start_time))
    # file.write('range%d   train_accur%.6f   train_loss%.6f\n' % (i, train_accur, train_los))
    # file.write('range%d   val_accur%.6f   val_loss%.6f\n\n' % (i, val_accur, val_los))
    sess.close()
    # iters = range(len(val_acc))
    # # plt.figure()
    # plt.subplot(2, 1, 1)
    # #
    # plt.plot(iters, train_acc, 'r', label='train acc')
    # #
    # plt.plot(iters, val_acc, 'b', label='val acc')
    # plt.grid(True)
    # plt.legend(loc="right")
    # plt.show()
    # plt.subplot(2, 1, 2)
    # # val_acc
    # plt.plot(iters, train_loss, 'g', label='train loss')
    # # val_loss
    # plt.plot(iters, val_loss, 'k', label='val loss')
    # plt.grid(True)
    # # plt.xlabel(loss_type)
    # plt.ylabel('acc-loss')
    # plt.legend(loc="upper right")
    # plt.show()
