from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import cv2
import time
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.regularizers import l2
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
weight_decay = 0.0005

# model = Sequential()
# model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_train.shape[1:]
#                  , kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.3))
#
# model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
# model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
# model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
# model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
# model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
# model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
# model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.4))
#
# model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))
#
# model.add(Flatten())
# model.add(Dense(512, kernel_regularizer=l2(weight_decay)))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
#
# model.add(Dropout(0.5))
#
# model.add(Dense(10))
# model.add(Activation('softmax'))


def normalize(X_train, X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = normalize(x_train, x_test)
start_time = time.time()
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)
opt = keras.optimizers.adam(0.0001)
# model.compile(loss='categorical_crossentropy',
#               optimizer=opt,
#               metrics=['accuracy'])
model = keras.models.load_model('my_vgg16.h5')
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=256),
                              steps_per_epoch=x_train.shape[0] // 256,
                              epochs=300, validation_data=(x_test, y_test), verbose=2)
model.save('my_vgg16.h5')
plt.plot()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
file = open('cifar10_result', 'a')
file.write(
    '\nepochs %d sec_cost %.2f model vgg16 adam lr=0.001 datagen\n' % ( \
        len(history.history['loss']), time.time() - start_time))
file.write('tra_acc%.4f   tra_loss%.6f\n' % (history.history['acc'][-1], history.history['loss'][-1]))
file.write('val_acc%.4f   val_loss%.6f\n' % (history.history['val_acc'][-1], history.history['val_loss'][-1]))
file.write(str(history.history)+'\n')
file.close()
