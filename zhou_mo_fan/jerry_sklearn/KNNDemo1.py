from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sklearn

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=0.3)
knn = KNeighborsClassifier()
knn.fit(train_x, train_y)
print(knn.predict(test_x))
print(test_y)
print(knn.get_params())
print(knn.score(iris_x,iris_y))