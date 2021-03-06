import tensorflow as tf

filter_weight = tf.get_variable('weight', [5, 5, 3, 16])
biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.1))
conv = tf.nn.conv2d(input, filter_weight, [1, 1, 1, 1], padding='SAME')
bias = tf.nn.bias_add(conv,biases)
actived_conv = tf.nn.relu(bias)
pool = tf.nn.max_pool(actived_conv,ksize=[1,3,3,1],
                      strides=[1,2,2,1],padding='SAME')

