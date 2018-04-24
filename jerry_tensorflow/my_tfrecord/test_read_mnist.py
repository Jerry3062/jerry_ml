import numpy as np
import pandas as pd
import tensorflow as tf

for serialized_example in tf.python_io.tf_record_iterator('F:/dataset/mnist/train.tfrecord'):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    image = example.features.feature['img_raw'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    print(image, label)
    pass
