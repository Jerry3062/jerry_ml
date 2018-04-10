# 合并array
import numpy as np

a = np.array([1, 1, 1])
b = np.array([2, 2, 2])
# vstack 上下合并
print(np.vstack((a, b)))
# hstack 水平合并
print(np.hstack((a, b)))
# 增加一个维度
print(a[:, np.newaxis])
# concatenate多个array的合并
c = np.concatenate((a,b),axis=0)
print(c)
