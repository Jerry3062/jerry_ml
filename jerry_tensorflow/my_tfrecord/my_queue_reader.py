import numpy as np
import pandas as np
import tensorflow as tf


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28])
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.cast(features['label'], tf.int32)
    return img, label


# img, label = read_and_decode('F:/dataset/mnist/train.tfrecord')
# img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=30, capacity=2000, min_after_dequeue=1000)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     threads = tf.train.start_queue_runners(sess=sess)
#     for i in range(3):
#         val, l = sess.run(img_batch, label_batch)
#         print(val.shape)
#         print(l)

filename_queue = tf.train.string_input_producer(['F:/dataset/mnist/train.tfrecord'])
reader = tf.TFRecordReader()
_, serizlized_example = reader.read(filename_queue)
features = tf.parse_single_example(serizlized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string)
                                   })
image = tf.decode_raw(features['img_raw'], tf.int64)
image = tf.reshape(image, [28, 28])
label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(20):
        example, l = sess.run([image, label])
        print(example.shape)
        print(l)
    coord.request_stop()
    coord.join(threads)
