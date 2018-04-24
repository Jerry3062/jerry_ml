import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from keras.utils.np_utils import to_categorical


def read_batchs():
    filename_queue = tf.train.string_input_producer(['F:/dataset/mnist/train.tfrecord'])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string)
    })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [784])
    label = tf.cast(features['label'], tf.int32)
    image_batchs, label_batchs = tf.train.shuffle_batch([image, label],
                                                        batch_size=128, capacity=100, min_after_dequeue=10,
                                                        num_threads=10)
    return image_batchs, label_batchs


with tf.device('/gpu:0'):
    learning_rate = 1e-4
    training_iters = 200
    batch_size = 200
    display_step = 5
    n_classes = 10
    n_fc1 = 120
    n_fc2 = 82

    # 构建模型
    input_x = tf.placeholder(tf.float32, [None, 784]) / 255.
    x = tf.reshape(input_x, [-1, 28, 28, 1])
    y = tf.placeholder(tf.int32, [None, n_classes])

    conv1 = tf.layers.conv2d(x, 6, [3, 3], 1, 'same', activation=tf.nn.relu)

    poo1 = tf.layers.max_pooling2d(conv1, [2, 2], 2)

    conv2 = tf.layers.conv2d(poo1, 16, [3, 3], 1, 'same', activation=tf.nn.relu)

    poo2 = tf.layers.max_pooling2d(conv2, [2, 2], 2)

    flat = tf.reshape(poo2, [-1, 7 * 7 * 16])
    dense = tf.layers.dense(flat, 120, activation=tf.nn.relu)

    dense2 = tf.layers.dense(dense, 84, activation=tf.nn.relu)

    logits = tf.layers.dense(dense2, 10)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1), predictions=tf.argmax(logits, axis=1))[1]

    image_batch, label_batch = read_batchs()

    # 创建会话
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    for i in range(10000):
        train_x, train_y = sess.run([image_batch, label_batch])
        train_y = to_categorical(train_y, 10)
        sess.run(train_op, feed_dict={input_x: train_x, y: train_y})
        print(sess.run(accuracy, feed_dict={input_x: train_x, y: train_y}))
    coord.request_stop()
    coord.join(threads)
