from __future__ import print_function
import numpy as np

np.random.seed(6666)

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import pandas as pd

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

batch_size = 1024
nb_classes = 10
nb_epoch = 10

img_rows, img_cols = 28, 28
nb_filters = 64
pool_size = (2, 2)
kernel_size = (3, 3)

train_origin = pd.read_csv("F:/dataset/mnist/train210000.csv", index_col=False)

test_data_x = pd.read_csv("F:/dataset/mnist/test.csv")
train_y = train_origin.label

train = train_origin.drop(['label'], axis=1)

train_y = train_y.as_matrix()
# 将label转为one_hot
new_train_y = []
for item in train_y:
    item_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    item_list[item] = 1
    new_train_y.append(item_list)
train = pd.concat((train, pd.DataFrame(new_train_y)), axis=1)
train = train.as_matrix()
np.random.shuffle(train)

X_train = train[:, :-10]
y_train = train[:, -10:]

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')

# Y_train = np_utils.to_categorical(y_train, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size, padding='same', input_shape=input_shape, name='conv1'))
model.add(BatchNormalization(name='bn1'))
model.add(Activation('relu'))
model.add(BatchNormalization(name='bn2'))

model.add(Convolution2D(nb_filters, kernel_size, padding='same', input_shape=input_shape, name='conv2'))
model.add(BatchNormalization(name='bn3'))
model.add(Activation('relu'))
model.add(BatchNormalization(name='bn4'))
model.add(MaxPooling2D(pool_size, name='pool1'))

model.add(Flatten(name='flat'))

model.add(Dense(64, name='fc1'))
model.add(BatchNormalization(name='bn5'))
model.add(Activation('relu'))
model.add(Dense(64, name='fc2'))
model.add(BatchNormalization(name='bn6'))
model.add(Activation('relu'))
model.add(Dense(10, name='fc3'))
model.add(BatchNormalization(name='bn7'))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1)

# score = model.evaluate(X_test, Y_test, verbose=0)

# model.save('cnn_model.h5')

test_x_data = pd.read_csv('F:/dataset/mnist/test.csv').as_matrix()
test_x_data = np.reshape(test_x_data, (-1, 28, 28, 1))
prediction = model.predict(test_x_data)
print(type(prediction))
submission = pd.DataFrame({'ImageId': np.arange(1, 28001), 'Label': np.argmax(prediction, axis=1)})
submission.to_csv("submission.csv", index=False)

# print('Test loss: {}'.format(score[0]))
# print('Test accuracy: {}'.format(score[1]))
