import tflearn
import tensorflow as tf
import numpy as np
import pandas as pd
import tflearn.datasets.mnist as mnist


# trainData = pd.read_csv("F:/dataset/mnist/train.csv")
# trainData = trainData.as_matrix()
# trainX = trainData[:, :-1]
# trainY = trainData[:, 0]
# trainY = np.atleast_2d(trainY)


trainX,trainY,TestX,testY = mnist.load_data(one_hot=True)
print(trainX.shape)
print(trainY.shape)

# Inputs
net = tflearn.input_data([None, trainX.shape[1]])
# Hidden layers
net = tflearn.fully_connected(net, 128, activation='ReLU')
net = tflearn.fully_connected(net, 32, activation='ReLU')

# Output layer and training model
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='sgd', learning_rate=0.01)

model = tflearn.DNN(net)

# Training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=100, n_epoch=100)

# 暂时无法运行
