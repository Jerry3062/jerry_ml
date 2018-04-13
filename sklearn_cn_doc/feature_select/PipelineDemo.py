# 在这段代码中，我们利用 sklearn.svm.LinearSVC 和
# sklearn.feature_selection.SelectFromModel 来评估特征的重要性并且选择出相关的特征。
#  然后，在转化后的输出中使用一个 sklearn.ensemble.RandomForestClassifier 分类器，
# 比如只使用相关的特征。你也可以使用其他特征选择的方法和可以提供评估特征重要性的
# 分类器来执行相似的操作。 请查阅 sklearn.pipeline.Pipeline 来了解更多的实例。

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC())), ('classification', GradientBoostingClassifier())])

iris = load_iris()
x = iris.data
y = iris.target
scores = cross_val_score(clf, x, y)
print(scores.mean())

# Pipeline对象接受二元tuple构成的list，每一个二元 tuple 中的第一个元素为 arbitrary identifier string，
# 我们用以获取（access）Pipeline object 中的 individual elements，二元 tuple
# 中的第二个元素是 scikit-learn与之相适配的transformer 或者 estimator。
