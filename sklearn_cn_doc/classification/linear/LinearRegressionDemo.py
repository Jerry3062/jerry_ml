from sklearn import linear_model

# LinearRegression 拟合一个带有系数 w = (w_1, ..., w_p) 的线性模型，
# 使得数据集实际观测数据和预测数据（估计值）之间的残差平方和最小

reg = linear_model.LinearRegression()
reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(reg.coef_)
print(reg.intercept_)

# 然而，对于普通最小二乘的系数估计问题，其依赖于模型各项的相互独立性。
# 当各项是相关的，且设计矩阵 X 的各列近似线性相关，那么，设计矩阵会趋向于奇异矩阵，
# 这会导致最小二乘估计对于随机误差非常敏感，产生很大的方差。例如，在没有实验设计
# 的情况下收集到的数据，这种多重共线性（multicollinearity）的情况可能真的会出现。
#
#该方法使用 X 的奇异值分解来计算最小二乘解。如果 X 是一个 size 为 (n, p) 的矩阵，设 n \geq p ，
# 则该方法的复杂度为 O(n p^2)