import numpy as np

# 生成0矩阵
a = np.zeros((3, 4), dtype=np.float32)

b = np.arange(10, 30, 2, dtype=np.int)
c = b.reshape((5, 2))
print(c)
# 0到10之间生成5个数
d = np.linspace(0, 10, 5)
print(d)