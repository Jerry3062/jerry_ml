import tensorflow as tf

filename_queue = tf.train.string_input_producer(['F:/dataset/mnist/train.tfrecord'])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features={
    'label': tf.FixedLenFeature([], tf.int64),
    'img_raw': tf.FixedLenFeature([], tf.string)
})
label = tf.cast(features['label'],tf.int32)
image = tf.decode_raw(features['img_raw'],tf.uint8)
image = tf.reshape(image,[28,28])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    x,y = sess.run([image,label])
    print(x,y)
    coord.request_stop()
    coord.join(threads)