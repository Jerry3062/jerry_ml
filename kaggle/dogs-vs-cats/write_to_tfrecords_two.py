import tensorflow as tf
import os
import cv2
import numpy as np

for root, dirs, files in os.walk('F:/dataset/dogs-vs-cats-resized/train/'):
    tf_writer = tf.python_io.TFRecordWriter('F:/dataset/dogs-vs-cats-resized/train.tfrecords')
    images = []
    labels = []
    for file in files:
        label = int(file.split('.')[0] == 'cat')
        imagepath = os.path.join(root, file)
        images.append(imagepath)
        labels.append(label)
    print(len(images))
    print(len(labels))
    all_data = [images, labels]
    all_data = np.array(all_data).T
    np.random.shuffle(all_data)
    print(all_data[0, 0])
    print(all_data[0, 1])
    for line in all_data:
        img_raw = cv2.imread(line[0]).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(line[1])])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        tf_writer.write(example.SerializeToString())
