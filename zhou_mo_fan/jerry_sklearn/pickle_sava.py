from sklearn import datasets
from sklearn import svm
import pickle

# clf = svm.SVC()
iris = datasets.load_iris()
x, y = iris.data, iris.target
# clf.fit(x, y)
#
# 将模型存到文件
# with open("clf.pickle", 'wb') as f:
#     pickle.dump(clf, f)
#     print('ok')

with open('clf.pickle', 'rb') as f:
    clf = pickle.load(f)
    print(clf.predict(x[:1]))
