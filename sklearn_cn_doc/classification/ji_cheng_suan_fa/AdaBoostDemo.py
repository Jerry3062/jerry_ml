from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf,iris.data,iris.target)
print(scores.mean())

