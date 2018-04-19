import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
from sklearn import preprocessing

# 加载mnist_inference.py中定义的常量和前向传播的函数
from jerry_tensorflow.book_tensorfolw_google_action.best_practice import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 100000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'model.ckpt'


def train(train_data, mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')/255.
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用mnist_inference.py中定义的前向传播过程
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    # 定义损失函数，学习率，滑动平均操作以及训练过程
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_average.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            if i % 2100 ==0:
                tf.random_shuffle(train_data)
                train_x = train_data[:, :-10]
                train_y = train_data[:, -10:]
            batch_x = train_x[i * 100:(i + 1) * 100]
            batch_y = train_y[i * 100:(i + 1) * 100]
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: batch_x, y_: batch_y})
            # 每10000轮保存一次模型
            if i % 30000 == 0:
                # 输出当前的训练情况，这里只输出了模型在当前训练batch上的损失函数大小。通过损失
                # 函数的大小可以大概了解训练的情况。在验证数据集上的正确信息会有一个单独的程序来生成
                print('After %d training steps,loss on training  is %g.' % (step, loss_value))
                # 保存当前的模型。注意这里给出了global_step参数，这样可以让每个被保存模型的文件名末尾加
                # 上训练的轮数，比如‘model.ckpt-1000'表示训练1000轮之后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    train_origin = pd.read_csv("F:/dataset/mnist/train210000.csv", index_col=False)
    train_y = train_origin.label
    train_data = train_origin.drop(['label'], axis=1)
    train_y = train_y.as_matrix()
    # 将label转为one_hot
    from keras.utils.np_utils import to_categorical
    train_y = to_categorical(train_y, num_classes=10)
    train_data = pd.concat((train_data, pd.DataFrame(train_y)), axis=1)
    train_data = train_data.as_matrix()
    mnist = input_data.read_data_sets('../../../MNIST_data', one_hot=True)
    train(train_data, mnist)


if __name__ == '__main__':
    tf.app.run()
