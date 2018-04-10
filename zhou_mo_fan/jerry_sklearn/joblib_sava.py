from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib
# clf = svm.SVC()
iris = datasets.load_iris()
x, y = iris.data, iris.target
# clf.fit(x, y)
#
# 将模型存到文件
# joblib.dump(clf, 'clf.joblib')

clf = joblib.load('clf.joblib')
print(clf.predict(x[:1]))

