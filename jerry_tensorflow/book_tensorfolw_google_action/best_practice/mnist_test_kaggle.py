import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import numpy as np

# 加载mnist_inference.py和mnist_train.py中定义的常量和函数
from jerry_tensorflow.book_tensorfolw_google_action.best_practice import mnist_inference
from jerry_tensorflow.book_tensorfolw_google_action.best_practice import mnist_train

test_x = pd.read_csv('F:/dataset/mnist/test.csv').as_matrix()
x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE])

y = mnist_inference.inference(x, None)
y = tf.argmax(y, 1)
variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
variables_ro_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_ro_restore)
with tf.Session() as sess:
    # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
    ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        # 加载模型
        saver.restore(sess, ckpt.model_checkpoint_path)
        # 通过文件名得到模型保存时迭代的轮数
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        accuracy_score = sess.run(y, feed_dict={x:test_x})
        submission = pd.DataFrame({'ImageId': np.arange(1, 28001), 'Label': accuracy_score})
        submission.to_csv("submission.csv", index=False)
    else:
        print('No checkpoint file found')
