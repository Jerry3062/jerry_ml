import os
import numpy as np
import tensorflow as tf
import cv2


def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))
        # print(files)
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('\\')[-1]
        if letter == 'cat':
            labels = np.append(labels, n_img * [0])
        else:
            labels = np.append(labels, n_img * [1])
    # shuffle
    # print(images[0])
    # print(len(labels))
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecord(images_list, label_list, save_dir, name):
    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(images_list)
    writer = tf.python_io.TFRecordWriter(filename)
    # print('Transform start-----------------------')
    error_count = 0
    for i in np.arange(0, n_samples):
        try:
            image = cv2.imread(images_list[i])
            image_raw = image.tostring()
            label = int(label_list[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': int64_feature(label),
                'image_raw': bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        except BaseException as e:
            # error_count += 1
            print(e)
            # print('Could not read:', images_list[i])
    print(error_count)
    writer.close()
    print('Transform done')


images_list, label_list = get_file('F:/dataset/dogs-vs-cats-resized')
convert_to_tfrecord(images_list, label_list, "F:/dataset/dogs-vs-cats-resized", 'train')
