from pengliang.jerry_nn.network2 import Network
from pengliang.jerry_nn import mnist_loader
import pandas as pd
import numpy as np


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = np.array(list(test_data))
print("training data")
print(type(training_data))
print(len(training_data))
print(training_data[0][0].shape)
print(training_data[0][1].shape)
# train = pd.read_csv('F:/dataset/mnist/train.csv')
test = pd.read_csv('F:/dataset/mnist/test.csv')
#
# train_data = train.as_matrix()[:32000]
# validation = train.as_matrix()[32000:]
test_data2 = test.as_matrix()
#
# print(train_data.shape)
# print(validation.shape)
# print(test_data.shape)
#
net = Network([784, 100,30,10])
net.SGD(training_data, 30, 100, 0.5, 2, evaluation_data=validation_data,
        monitor_evaluation_accuracy=True, monitor_training_cost=True,
        monitor_training_accuracy=True, monitor_evaluation_cost=True)
predictions = net.predict(test_data2)
submission = pd.DataFrame({'ImageId': np.arange(1, 28001), 'Label': predictions})
submission.to_csv("submission.csv", index=False)
