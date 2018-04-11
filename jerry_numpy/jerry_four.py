import numpy as np

a = np.random.random((2, 4))
print(a)

# axis 在那个维度中计算
print(np.sum(a, axis=0))
print(np.min(a, axis=1))
print(np.max(a, axis=0))

b = np.arange(2, 14).reshape((3, 4))
# argmax 取最大值的索引
print(np.argmax(b))
# 取平均
print(b.mean())
# 取中位数
print(np.median(b))
# cumsum 就像reduce那样相加
print(np.cumsum(b))
# diff相邻数字的差
print(np.diff(b))
# clip 将超边界值变为边界值
print(np.clip(b, 5, 9))
print(b)
print(np.mean(b, axis=0))
