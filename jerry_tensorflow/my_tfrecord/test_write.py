import pandas as pd
import numpy as np
import tensorflow as tf

train = pd.read_csv('F:/dataset/mnist/train.csv')
y = train.label
x = train.drop(['label'], axis=1)
del train
x = x.as_matrix()
y = y.as_matrix()

filepath = 'F:/dataset/mnist/train.tfrecord'
writer = tf.python_io.TFRecordWriter(filepath)
for i in range(len(x)):
    label = y[i]
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[x[i].tobytes()]))
    }))
    writer.write(example.SerializeToString())
writer.close()
