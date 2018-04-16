from pengliang.jerry_nn.network3 import Network
from pengliang.jerry_nn import network3
from pengliang.jerry_nn.network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from pengliang.jerry_nn import mnist_loader
import pandas as pd
import numpy as np

training_data, validation_data, test_data = network3.load_data_shared()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = np.array(list(test_data))
expanded_training_data, _, _ = network3.load_data_shared("F:/dataset/mnist/mnist_expanded.pkl.gz")
mini_batch_size = 10
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                             filter_shape=(20, 1, 5, 5),
                             poolsize=(2, 2),
                             activation_fn=network3.ReLU),
               ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                             filter_shape=(40, 20, 5, 5),
                             poolsize=(2, 2),
                             activation_fn=network3.ReLU),
               FullyConnectedLayer(n_in=40 * 4 * 4, n_out=100, activation_fn=network3.ReLU),
               FullyConnectedLayer(n_in=100 * 4 * 4, n_out=100, activation_fn=network3.ReLU),
               SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)]
              , mini_batch_size)
test = pd.read_csv('F:/dataset/mnist/test.csv')
net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)

# test_data2 = test.as_matrix()
#
#
# predictions = net.predict(test_data2)
# submission = pd.DataFrame({'ImageId': np.arange(1, 28001), 'Label': predictions})
# submission.to_csv("submission.csv", index=False)
