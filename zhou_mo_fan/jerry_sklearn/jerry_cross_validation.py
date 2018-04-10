from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
x = iris.data
y = iris.target
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn,x,y,cv=5,verbose=1,scoring='accuracy')
print(scores)
print(scores.mean())