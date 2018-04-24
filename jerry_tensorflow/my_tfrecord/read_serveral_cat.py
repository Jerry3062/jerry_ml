import tensorflow as tf

filename_queue = tf.train.string_input_producer(['F:/dataset/test-cat/cats.records'])
reader = tf.TFRecordReader()
_, serizalized_example = reader.read(filename_queue)
features = tf.parse_single_example(serizalized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'image': tf.FixedLenFeature([], tf.string)
                                   })
image = tf.decode_raw(features['image'], tf.uint8)
image = tf.reshape(image, [277, 277, 3])
label = tf.cast(features['label'], tf.int32)
image_batch, label_batch = tf.train.batch([image, label], batch_size=5, capacity=10)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    img, lab = sess.run([image_batch, label_batch])
    # img, lab = sess.run([image, label])
    print(img.shape, lab)
    coord.request_stop()
    coord.join(threads)
