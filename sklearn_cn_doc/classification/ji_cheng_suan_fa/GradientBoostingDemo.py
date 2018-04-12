from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()
clf = GradientBoostingClassifier()
scores = cross_val_score(clf,iris.data,iris.target,cv=10)
print(scores.mean())

#超过两类的分类问题需要在每一次迭代时推导
# n_classes 个回归树。因此，所有的需要推导的树数量等于
# n_classes * n_estimators 。对于拥有大量类别的数据集
# 我们强烈推荐使用 RandomForestClassifier 来代替
#  GradientBoostingClassifier 。