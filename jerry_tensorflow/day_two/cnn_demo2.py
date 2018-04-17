import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import random
import pandas as pd

train_origin = pd.read_csv("F:/dataset/mnist/train210000.csv",index_col=False)
print(train_origin.shape)
# train_left = pd.read_csv("F:/dataset/mnist/train_left.csv",index_col=False)
# print(train_left.shape)
# train_right = pd.read_csv("F:/dataset/mnist/train_right.csv",index_col=False)
# print(train_right.shape)
# train_top = pd.read_csv("F:/dataset/mnist/train_top.csv",index_col=False)
# print(train_top.shape)
# train_bottom = pd.read_csv("F:/dataset/mnist/train_bot.csv",index_col=False)
# print(train_bottom.shape)
# train = pd.concat((train_origin, train_left),axis=0,ignore_index=True)
# print(train.shape)
# train = pd.concat((train, train_right), axis=0, ignore_index=True)
# print(train.shape)
# train = pd.concat((train, train_top), axis=0, ignore_index=True)
# print(train.shape)
# train = pd.concat((train, train_bottom), axis=0, ignore_index=True)
# train.to_csv('F:/dataset/mnist/train210000.csv',index=False)
#
# test = pd.read_csv("F:/dataset/mnist/test.csv")
# train_y = train_origin.label
# train = train_origin.drop(['label'], axis=1)
# train_y = train_y.as_matrix()
# # 将label转为one_hot
# new_train_y = []
# for item in train_y:
#     item_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     item_list[item] = 1
#     new_train_y.append(item_list)
# train = pd.concat((train,pd.DataFrame(new_train_y)),axis=1)

# left_train = train.as_matrix().copy()
#
# # 往左移动一个像素点
# new_left = []
# for line in left_train:
#     line = np.reshape(line, (28, 28))
#     line = line[:, 1:]
#     add_left = [0 for i in range(28)]
#     add_left = np.asarray(add_left)
#     add_left = np.reshape(add_left, (28, 1))
#     line = pd.DataFrame(line)
#     add_left = pd.DataFrame(add_left)
#     line = pd.concat((line, add_left), axis=1)
#     new_left.append(np.reshape(line.as_matrix(), (784,)))
# print(np.asarray(new_left).shape)
# new_left = pd.DataFrame(np.asarray(new_left))
# new_left = pd.concat((train_y, new_left),axis=1,ignore_index=True)
# print(new_left.shape)
# new_left.to_csv("F:/dataset/mnist/train_left.csv",index=False)

# 往右移动一个像素点
# right_train = train.as_matrix().copy()
# new_right = []
# for line in right_train:
#     line = np.reshape(line, (28, 28))
#     line = line[:, :-1]
#     add_right = [0 for i in range(28)]
#     add_right = np.asarray(add_right)
#     add_right = np.reshape(add_right, (28, 1))
#     line = pd.DataFrame(line)
#     add_right = pd.DataFrame(add_right)
#     line = pd.concat((add_right, line), axis=1)
#     new_right.append(np.reshape(line.as_matrix(), (784,)))
#
# print(np.asarray(new_right).shape)
# new_right = pd.DataFrame(np.asarray(new_right))
# new_right = pd.concat((train_y, new_right),axis=1)
# print(new_right.shape)
# new_right.to_csv("F:/dataset/mnist/train_right.csv",index=False)


# # 往上移动一个像素点
# top_train = train.as_matrix().copy()
# new_top = []
# for line in top_train:
#     line = np.reshape(line, (28, 28))
#     line = line[1:]
#     add_top = [0 for i in range(28)]
#     add_top = np.asarray(add_top)
#     add_top = np.reshape(add_top, (1, 28))
#     line = pd.DataFrame(line)
#     add_top = pd.DataFrame(add_top)
#     line = pd.concat((line, add_top), axis=0)
#     new_top.append(np.reshape(line.as_matrix(), (784,)))
#
# print(np.asarray(new_top).shape)
# new_top = pd.DataFrame(np.asarray(new_top))
# new_top = pd.concat((train_y, new_top),axis=1)
# print(new_top.shape)
# new_top.to_csv("F:/dataset/mnist/train_top.csv",index=False)
# # 往下移动一个像素点
# bot_train = train.as_matrix().copy()
# new_bot = []
# for line in bot_train:
#     line = np.reshape(line, (28, 28))
#     line = line[:-1]
#     add_bot = [0 for i in range(28)]
#     add_bot = np.asarray(add_bot)
#     add_bot = np.reshape(add_bot, (1, 28))
#     line = pd.DataFrame(line)
#     add_bot = pd.DataFrame(add_bot)
#     line = pd.concat((add_bot, line), axis=0)
#     new_bot.append(np.reshape(line.as_matrix(), (784,)))
# print(np.asarray(new_bot).shape)
# print(np.asarray(new_bot).shape)
# new_bot = pd.DataFrame(np.asarray(new_bot))
# new_bot = pd.concat((train_y, new_bot),axis=1)
# print(new_bot.shape)
# new_bot.to_csv("F:/dataset/mnist/train_bot.csv",index=False)

# input_x = tf.placeholder(tf.float32, [None, 784]) / 255.
# output_y = tf.placeholder(tf.float32, [None, 10])
# input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])
#
# test_x = train.as_matrix()
# test_y = new_train_y
#
# # 构建卷积神经网络
# # 第一层卷积
# conv1 = tf.layers.conv2d(
#     inputs=input_x_images,  # 形状[28,28,1]
#     filters=32,  # 32个过滤器，输出的深度是32
#     kernel_size=[5, 5],  # 过滤器在二维的大小是（5*5)
#     strides=1,  # 步长是1
#     padding='same',  # same表示输出的大小不变，在外围补0两圈，‘valid'不过超过边界
#     activation=tf.nn.relu,  # 激活函数是Relu
#     name='conv1'
# )
#
# pool1 = tf.layers.max_pooling2d(
#     inputs=conv1,  # 形状[14,14,32]
#     pool_size=[2, 2],  # 过滤器在二维的大小是（2*2）
#     strides=2,  # 步长2
#     name='pool1'
# )
#
# conv2 = tf.layers.conv2d(
#     inputs=pool1,  # 形状[28,28,1]
#     filters=64,  # 64个过滤器，输出的深度是64
#     kernel_size=[5, 5],  # 过滤器在二维的大小是（5*5)
#     strides=1,  # 步长是1
#     padding='same',  # same表示输出的大小不变，在外围补0两圈，‘valid'不过超过边界
#     activation=tf.nn.relu,  # 激活函数是Relu
#     name='conv2'
# )
#
# pool2 = tf.layers.max_pooling2d(
#     inputs=conv2,
#     pool_size=[2, 2],  # 过滤器在二维的大小是（2*2）
#     strides=2,  # 步长2
#     name='pool2'
# )
#
# # 平坦化
# flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
# # 1024 个神经元的全连接层
# dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
#
# # Dropout ：丢弃50%
# dropout = tf.layers.dropout(inputs=dense, rate=0.5)
# # 10个神经元的全连接层，这里不用激活函数来做非线性化了 输出形状【1,1,10】
# logits = tf.layers.dense(inputs=dropout, units=10)
#
# # 计算误差 （计算Cross entropy（交叉熵），再用softmax计算百分比概率
# loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)
#
# # 用 Adam优化器来最小化误差
# train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#
# # 计算预测值和实际标签的匹配程度
# # 此方法返回（accuracy,update_op）会创建两个局部变量
# accuracy = tf.metrics.accuracy(labels=tf.argmax(output_y, axis=1),
#                                predictions=tf.argmax(logits, axis=1))[1]
# # 创建会话
# sess = tf.Session()
# # 初始化变量，全局和局部
# init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# sess.run(init)
# for i in range(20000):
#     item_index = i % 420
#     batch_x = test_x[item_index*100:(item_index+1)*100]
#     batch_y = test_y[item_index*100:(item_index+1)*100]
#     train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch_x, output_y:batch_y })
#     if i % 1000 == 0:
#         test_accuracy = sess.run(accuracy, {input_x: test_x[:10000], output_y: test_y[:10000]})
#         print("Step%d,Train loss=%.7f,[Test accuracy=%.5f]" % (i, train_loss, test_accuracy))
# sess.close()
