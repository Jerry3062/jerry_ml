import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../day_one/MNIST-data', one_hot=True)
# a = mnist.train.next_batch(1)[0]
# b = np.reshape(a,(28,28))
# plt.imshow(b)
# plt.show()
# print(b.shape)
input_x = tf.placeholder(tf.float32, [None, 784]) / 255.
output_y = tf.placeholder(tf.float32, [None, 10])
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

# 从Test数据里选取3000个手写数字的图片和对应标签
test_x = mnist.test.images
test_y = mnist.test.labels

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

# 用 Adam优化器来最小化误差
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

# 计算预测值和实际标签的匹配程度
# 此方法返回（accuracy,update_op）会创建两个局部变量
accuracy = tf.metrics.accuracy(labels=tf.argmax(output_y, axis=1),
                               predictions=tf.argmax(logits, axis=1))[1]
# 创建会话
sess = tf.Session()
# 初始化变量，全局和局部
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)
for i in range(200000):
    batch = mnist.train.next_batch(100)
    train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
    if i % 1000 == 0:
        test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
        print("Step%d,Train loss=%.7f,[Test accuracy=%.5f]" % (i, train_loss, test_accuracy))
    # test_output = sess.run(logits, {input_x: test_x[:20]})
    # inferenced_y = np.argmax(test_output, 1)
sess.close()
    # print(inferenced_y, 'Inferenced numbers')
    # print(np.argmax(test_y[:20], 1), 'Real numbers')
