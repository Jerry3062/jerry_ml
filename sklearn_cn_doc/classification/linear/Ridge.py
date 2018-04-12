# 岭回归Ridge 回归通过对系数的大小施加惩罚来解决 普通最小二乘法 的一些问题
from sklearn import linear_model

# reg = linear_model.Ridge(alpha=.5)
# reg.fit([[0,0],[0,0],[1,1]],[0,.1,1])
# print(reg.coef_)
# print(reg.intercept_)

# RidgeCV 通过内置的 Alpha 参数的交叉验证来实现岭回归。 该对象与 GridSearchCV
# 的使用方法相同，只是它默认为 Generalized Cross-Validation(广义交叉验证 GCV)，
# 这是一种有效的留一验证方法（LOO-CV）:
reg = linear_model.RidgeCV(alphas=[0.01, 0.1, 1, 10])
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print(reg.alpha_)
