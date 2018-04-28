from keras.models import Model
from keras.layers import Input, Dense, Dropout, \
    BatchNormalization, Conv2D, MaxPooling2D, Activation, ZeroPadding2D
from keras.layers import add, Flatten, AveragePooling2D
from keras.utils import plot_model, to_categorical
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import time
import os

# Global Constants
NB_CLASS = 10
IM_WIDTH = 32
IM_HEIGHT = 32
batch_size = 32


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


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1),
              padding='same'):
    x = Conv2D(nb_filter, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')(x)
    x = BatchNormalization()(x)
    return x


def identity_block(input, nb_filter, kernel_size, strides=(1, 1),
                   with_conv_shortcut=False):
    x = Conv2d_BN(input, nb_filter, kernel_size, strides, padding='same')
    x = Conv2d_BN(x, nb_filter, kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(input, nb_filter, strides=strides,
                             kernel_size=kernel_size,padding='same')
        x = add([x, shortcut])
    else:
        x = add([x, input])
    return x


def bottleneck_bock(input, nb_filters, strides=(1, 1), with_conv_stortcut=False):
    k1, k2, k3 = nb_filters
    x = Conv2d_BN(input, nb_filters=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filters=k2, kernel_size=3, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filters=k3, kernel_size=1, strides=strides, padding='same')
    if with_conv_stortcut:
        shortcut = Conv2d_BN(input, nb_filters=k3, strides=strides, kernel_size=1,padding='same')
        x = add([x, shortcut])
        return x
    else:
        x = add([x, input])
        return x


def resnet_34(width, height, channel, classes):
    input = Input(shape=(width, height, channel))
    # x = ZeroPadding2D((3,3))(input)
    # conv1
    x = Conv2d_BN(input, nb_filter=64, kernel_size=(3, 3), strides=(1, 1), padding='same')
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # conv2_x
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=64, kernel_size=(3, 3))

    # conv3_x
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=128, kernel_size=(3, 3))

    # conv4_x
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=256, kernel_size=(3, 3))

    # conv5_x
    x = identity_block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_block(x, nb_filter=512, kernel_size=(3, 3))
    x = identity_block(x, nb_filter=512, kernel_size=(3, 3))

    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    return model


def check_print():
    model = resnet_34(IM_WIDTH, IM_HEIGHT, 3, NB_CLASS)
    model.summary()
    # plot_model(model, 'resnet.png')
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Model Compoled')
    return model


if __name__ == '__main__':
    if os.path.exists('resnet_34.h5'):
        model = load_model('resnet_34.h5')
    else:
        model = check_print()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)
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
    start_time = time.time()
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=256),
                                  steps_per_epoch=x_train.shape[0] // 256,
                                  epochs=300, validation_data=(x_test, y_test),
                                  verbose=1)
    model.save('resnet34_cifar10.h5')
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
    file = open('cifar10_resnet', 'a')
    file.write(
        '\nepochs %d sec_cost %.2f model vgg19 momentum decay 0.914 ** (epoch // 3) datagen datagen\n' % ( \
            len(history.history['loss']), time.time() - start_time))
    file.write('tra_acc%.4f   tra_loss%.6f\n' % (history.history['acc'][-1], history.history['loss'][-1]))
    file.write('val_acc%.4f   val_loss%.6f\n' % (history.history['val_acc'][-1], history.history['val_loss'][-1]))
    file.write(str(history.history) + '\n')
    file.close()
