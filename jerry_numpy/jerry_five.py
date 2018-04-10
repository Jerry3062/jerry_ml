import numpy as np

a = np.arange(3, 15).reshape((3,4))
print(a)
print(a[2, 1])
#flatten 压扁成一行的序列
print(a.flatten())
#flat返回迭代器
for item in a.flat:
    print(item)
