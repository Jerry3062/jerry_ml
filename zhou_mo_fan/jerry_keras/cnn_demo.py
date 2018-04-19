import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.models import load_model

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)

KTF.set_session(sess)

# build CNN
model = Sequential()
# Conv layer 1 output shape(32,28,28)
model.add(Convolution2D(
    filters=32,
    kernel_size=(5, 5),
    padding='same',
    input_shape=(1, 28, 28),
))
model.add(Activation('relu'))
# Pooling layer 1 (max pooling) output shape(32,14,14)
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same',
))

# Conv layer 2 output shape(64,14,14)
model.add(Convolution2D(
    filters=64,
    kernel_size=(5, 5),
    padding='same'
))
model.add(Activation('relu'))
# Pooling layer 2 output shape(64,7,7)
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Fully connected layer 1 input shape(64*7*7)=(3136),output shape (1024)
model.add(Flatten())
model.add(Dense(1024))

model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
adam = Adam(lr=1e-4)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Training------------')
model.fit(X_train, y_train, nb_epoch=2, batch_size=100)
print('Testing----------------')
loss, accuracy = model.evaluate(X_test, y_test)
# model.save('cnn_model')
print('test loss', loss)
print('test accuracy', accuracy)

# model = load_model('cnn_model')
# print(model)
