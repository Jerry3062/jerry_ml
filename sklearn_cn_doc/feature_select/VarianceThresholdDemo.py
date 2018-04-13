# VarianceThreshold 是特征选择的一个简单基本方法，它会移除所有那些方差不满足一些阈
# 值的特征。默认情况下，它将会移除所有的零方差特征，即那些在所有的样本上的取值均不变的特征。
from sklearn.feature_selection import VarianceThreshold

x = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=0.2)
sel.fit(x)
print(x)
