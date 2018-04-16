import pandas as pd
import numpy as np
import tensorflow
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.info())