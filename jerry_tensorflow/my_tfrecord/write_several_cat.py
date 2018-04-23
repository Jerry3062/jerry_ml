import tensorflow as tf
import cv2
import numpy as np
import os

images = []
labels = []
for root, sub_folders, files in os.walk('F:/dataset/test-cat/'):
    for name in files:
        images.append(os.path.join(root, name))
        labels.append(int(name.split(".")[0] == 'cat'))

writer = tf.python_io.TFRecordWriter('F:/dataset/test-cat/cats.records')
for i in range(len(images)):
    imagepath = images[i]
    image = cv2.imread(imagepath)
    label = labels[i]
    image = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
    }))
    writer.write(example.SerializeToString())
writer.close()
