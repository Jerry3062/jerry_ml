import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
with tf.device('/gpu:1'):
    c = a + b
# 通过log_device_placement参数来输出运行每一个运算的设备
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))