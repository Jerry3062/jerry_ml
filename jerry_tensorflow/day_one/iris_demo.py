import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution

# print('Tensorflow version', tf.VERSION)
# print('Eager excution', tf.executing_eagerly())
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url), origin=train_dataset_url)
print('Local copy of the dataset file: {}'.format(train_dataset_fp))
