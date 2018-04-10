import numpy as np

a = np.array([10, 20, 30, 40])
b = np.arange(4)
print(a, b)
b = b.transpose()
print(np.dot(b, a))
print(a.dot(b))
print(b < 3)
