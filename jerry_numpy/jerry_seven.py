import numpy as np

a = np.arange(12).reshape((3, 4))
print(a)
# 分割  第二个参数分成多少份 第三个参数 在那个维度分
print(np.split(a, 2, 1))
#纵向分割
print(np.vsplit(a,3))
#横向分割
print(np.hsplit(a,2))
