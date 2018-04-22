
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def read_and_decode(tf_records_file, batch_size):
    filename_queue = tf.train.string_input_producer([tf_records_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      min_after_dequeue=100,
                                                      num_threads=64,
                                                      capacity=200)
    return image_batch, tf.reshape(label_batch, [batch_size])

batch ,data = read_and_decode('F:/dataset/dogs-vs-cats-resized/train.tfrecords',100)
print(type(batch),type(data))


# def onehot(labels):
#     n_sample = len(labels)
#     n_class = max(labels) + 1
#     onehot_labels = np.zeros((n_sample, n_class))
#     onehot_labels[np.arange(n_sample), labels] = 1
#     return onehot_labels
#
#
# learning_rate = 1e-4
# training_iters = 200
# batch_size = 50
# display_step = 5
# n_classes = 2
# n_fc1 = 4096
# n_fc2 = 2048
#
# x = tf.placeholder(tf.float32, [None, 227, 227, 3])
# y = tf.placeholder(tf.int32, [None, n_classes])
# W_conv = {
#     'conv1': tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.0001)),
#     'conv2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01)),
#     'conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01)),
#     'conv4': tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01)),
#     'conv5': tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01)),
#     'fc1': tf.Variable(tf.truncated_normal([13 * 13 * 256, n_fc1], stddev=0.1)),
#     'fc2': tf.Variable(tf.truncated_normal([n_fc1, n_fc2], stddev=0.1)),
#     'fc3': tf.Variable(tf.truncated_normal([n_fc2, n_classes], stddev=0.1)),
# }
# b_conv = {
#     'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[96])),
#     'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
#     'conv3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
#     'conv4': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
#     'conv5': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[256])),
#     'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc1])),
#     'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_fc2])),
#     'fc3': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_classes])),
# }

# x_image = tf.reshape(x, [-1, 227, 227, 3])
#
# # 第一层卷积池化层
# conv1 = tf.nn.conv2d(x_image, W_conv['conv1'], strides=[1, 4, 4, 1], padding='VALID')
# conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
# conv1 = tf.nn.relu(conv1)
#
# pool1 = tf.nn.avg_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
# norm1 = tf.nn.lrn(pool1, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
#
# # 第二层卷积池化层
# conv2 = tf.nn.conv2d(norm1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
# conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
# conv2 = tf.nn.relu(conv2)
#
# pool2 = tf.nn.avg_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
# norm2 = tf.nn.lrn(pool2, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
#
# # 第三层卷积层
# conv3 = tf.nn.conv3d(norm2, W_conv['conv3'], strides=[1, 1, 1, 1], padding='SAME')
# conv3 = tf.nn.bias_add(conv3, b_conv['conv3'])
# conv3 = tf.nn.relu(conv3)
#
# # 第四层卷积层
# conv4 = tf.nn.conv4d(conv3, W_conv['conv4'], strides=[1, 1, 1, 1], padding='SAME')
# conv4 = tf.nn.bias_add(conv4, b_conv['conv4'])
# conv4 = tf.nn.relu(conv4)
#
# # 第五层卷积层
# conv5 = tf.nn.conv5d(conv4, W_conv['conv5'], strides=[1, 1, 1, 1], padding='SAME')
# conv5 = tf.nn.bias_add(conv5, b_conv['conv5'])
# conv5 = tf.nn.relu(conv5)
#
# # 池化层
# pool5 = tf.nn.avg_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
#
# reshape = tf.reshape(pool5, [-1, 13, 13, 256])
#
# fc1 = tf.add(tf.matmul(reshape,W_conv['fc1']),b_conv['fc1'])
# fc1 = tf.nn.relu(fc1)
# fc1 = tf.nn.dropout(fc1,0.5)
#
# fc2 = tf.add(tf.matmul(fc1,W_conv['fc2']),b_conv['fc2'])
# fc2 = tf.nn.relu(fc2)
# fc2 = tf.nn.dropout(fc2,0.5)
#
# fc3 = tf.add(tf.matmul(fc2,W_conv['fc3']),b_conv['fc3'])
#
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc3,y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#
# correct_pred = tf.equal(tf.argmax(fc3,1),tf.argmax(y,1))
# accuracy  = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
# init = tf.global_variables_initializer()
#
# def train(opech):
#     with tf.Session() as sess:
#         sess.run(init)
#
#         train_writer = tf.summary.FileWriter(".//log", sess.graph)  # 输出日志的地方
#         saver = tf.train.Saver()
#
#         c = []
#         start_time = time.time()
#
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         step = 0
#         for i in range(opech):
#             step = i
#             image, label = sess.run([image_batch, label_batch])
#
#             labels = onehot(label)
#
#             sess.run(optimizer, feed_dict={x: image, y: labels})
#             loss_record = sess.run(loss, feed_dict={x: image, y: labels})
#             print("now the loss is %f " % loss_record)
#
#             c.append(loss_record)
#             end_time = time.time()
#             print('time: ', (end_time - start_time))
#             start_time = end_time
#             print("---------------%d onpech is finished-------------------" % i)
#         print("Optimization Finished!")
#         saver.save(sess, save_model)
#         print("Model Save Finished!")
#
#         coord.request_stop()
#         coord.join(threads)
#         plt.plot(c)
#         plt.xlabel('Iter')
#         plt.ylabel('loss')
#         plt.title('lr=%f, ti=%d, bs=%d' % (learning_rate, training_iters, batch_size))
#         plt.tight_layout()
#         plt.savefig('cat_and_dog_AlexNet.jpg', dpi=200)
#
# from PIL import Image
#
# def per_class(imagefile):
#
#     image = Image.open(imagefile)
#     image = image.resize([227, 227])
#     image_array = np.array(image)
#
#     image = tf.cast(image_array,tf.float32)
#     image = tf.image.per_image_standardization(image)
#     image = tf.reshape(image, [1, 227, 227, 3])
#
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#
#         save_model =  tf.train.latest_checkpoint('.//model')
#         saver.restore(sess, save_model)
#         image = tf.reshape(image, [1, 227, 227, 3])
#         image = sess.run(image)
#         prediction = sess.run(fc3, feed_dict={x: image})
#
#         max_index = np.argmax(prediction)
#         if max_index==0:
#             return "cat"
#         else:
#             return "dog"
