import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random
import pandas as pd

train_origin = pd.read_csv("F:/dataset/mnist/train210000.csv", index_col=False)

test_data_x = pd.read_csv("F:/dataset/mnist/test.csv")
train_y = train_origin.label

train = train_origin.drop(['label'], axis=1)

train_y = train_y.as_matrix()
# 将label转为one_hot
new_train_y = []
for item in train_y:
    item_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    item_list[item] = 1
    new_train_y.append(item_list)
train = pd.concat((train, pd.DataFrame(new_train_y)), axis=1)
train = train.as_matrix()

input_x = tf.placeholder(tf.float32, [None, 784]) / 255.
output_y = tf.placeholder(tf.float32, [None, 10])
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

# 构建卷积神经网络
# 第一层卷积
conv1 = tf.layers.conv2d(
    inputs=input_x_images,  # 形状[28,28,1]
    filters=32,  # 32个过滤器，输出的深度是32
    kernel_size=[5, 5],  # 过滤器在二维的大小是（5*5)
    strides=1,  # 步长是1
    padding='same',  # same表示输出的大小不变，在外围补0两圈，‘valid'不过超过边界
    activation=tf.nn.relu,  # 激活函数是Relu
    name='conv1'
)

pool1 = tf.layers.max_pooling2d(
    inputs=conv1,  # 形状[14,14,32]
    pool_size=[2, 2],  # 过滤器在二维的大小是（2*2）
    strides=2,  # 步长2
    name='pool1'
)

conv2 = tf.layers.conv2d(
    inputs=pool1,  # 形状[28,28,1]
    filters=64,  # 64个过滤器，输出的深度是64
    kernel_size=[5, 5],  # 过滤器在二维的大小是（5*5)
    strides=1,  # 步长是1
    padding='same',  # same表示输出的大小不变，在外围补0两圈，‘valid'不过超过边界
    activation=tf.nn.relu,  # 激活函数是Relu
    name='conv2'
)

pool2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=[2, 2],  # 过滤器在二维的大小是（2*2）
    strides=2,  # 步长2
    name='pool2'
)

# 平坦化
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
# 1024 个神经元的全连接层
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

# Dropout ：丢弃50%
dropout = tf.layers.dropout(inputs=dense, rate=0.5)
# 10个神经元的全连接层，这里不用激活函数来做非线性化了 输出形状【1,1,10】
logits = tf.layers.dense(inputs=dropout, units=10)

# 计算误差 （计算Cross entropy（交叉熵），再用softmax计算百分比概率
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

# global_step = tf.Variable(0, trainable=False)
# learning_rate = tf.train.exponential_decay(0.05, global_step, 2100, 0.96)

# 用 Adam优化器来最小化误差
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

# 计算预测值和实际标签的匹配程度
# 此方法返回（accuracy,update_op）会创建两个局部变量
accuracy = tf.metrics.accuracy(labels=tf.argmax(output_y, axis=1),
                               predictions=tf.argmax(logits, axis=1))[1]
prediction = tf.argmax(logits, axis=1)
# 创建会话
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内存
config.allow_soft_placement = True
sess = tf.Session(config=config)
# 初始化变量，全局和局部
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
np.random.shuffle(train)
validate_data = train[:10000]
validate_x = validate_data[:, :-10]
validate_y = validate_data[:, -10:]
train = train[10000:]
for i in range(600000):
    item_index = i % 2000
    if item_index == 0:
        np.random.shuffle(train)
        test_x = train[:, :-10]
        test_y = train[:, -10:]
    batch_x = test_x[item_index * 100:(item_index + 1) * 100]
    batch_y = test_y[item_index * 100:(item_index + 1) * 100]
    train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch_x, output_y: batch_y})
    if i % 1000 == 0:
        test_accuracy = sess.run(accuracy, {input_x: validate_x, output_y: validate_y})
        print("Step%d,Train loss=%.10f,[Test accuracy=%.5f]" % (i, train_loss, test_accuracy))

predictions = np.asarray([], dtype=np.int64)
for i in range(2800):
    batch_x = test_data_x[i * 100:(i + 1) * 100]
    prediction_data = sess.run(prediction, feed_dict={input_x: batch_x})
    predictions = np.hstack((predictions, prediction_data))
submission = pd.DataFrame({'ImageId': np.arange(1, 28001), 'Label': predictions})
submission.to_csv("submission.csv", index=False)

sess.close()
