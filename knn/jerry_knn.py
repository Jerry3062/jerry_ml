from sklearn import neighbors
from sklearn import tree
import sklearn
from sklearn import datasets
import jerry_numpy as np

knn = neighbors.KNeighborsClassifier()
tree = tree.DecisionTreeClassifier()
iris = datasets.load_iris()
print(sklearn.__version__)
# print(iris)
# print(type(iris))
# print(np.shape(iris))
knn.fit(iris.data ,iris.target)
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print(predictedLabel)
predictedLabel = tree.fit(iris.data,iris.target)
predictedLabel = knn.predict([[0.1, 3, 4, 0.4]])
print(predictedLabel)
from sklearn import svm
func = svm.SVC(kernel="linear")
func.fit(iris.data,iris.target)
print(func.predict([[0.1,3,4,0.4]]))

