import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

train_origin = pd.read_csv("F:/dataset/mnist/train210000.csv", index_col=False)
label = train_origin.label
train = train_origin.drop(['label'],axis=1)
one = train.as_matrix()[200000]
print(one)
one_label = label.as_matrix()[200000]
print(one_label)
one = np.reshape(one,(28,28))
sns.countplot(label)
plt.show()
