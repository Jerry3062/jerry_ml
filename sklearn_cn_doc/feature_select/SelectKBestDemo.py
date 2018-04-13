from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

iris = load_iris()
x, y = iris.data, iris.target
x_new = SelectKBest(chi2, 2).fit_transform(x,y)
print(x_new.shape)
