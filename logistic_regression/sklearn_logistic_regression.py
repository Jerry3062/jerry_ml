from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import jerry_numpy as np

iris = datasets.load_iris()
x = iris.data
y = iris.target
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
lr = LogisticRegression(solver='liblinear')
lr.fit(train_x,train_y)
predict_y = lr.predict(test_x)
print(test_y)
print(np.sum((test_y-predict_y)**2)/len(predict_y)*2)