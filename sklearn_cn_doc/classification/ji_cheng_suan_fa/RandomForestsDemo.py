from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
random_forest = RandomForestClassifier()
scores = cross_val_score(random_forest,iris.data,iris.target)
print(scores.mean())
