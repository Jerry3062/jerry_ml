import numpy as np

a = np.arange(4)
b = a.copy()  # deep copy
a[0] = 11
print(b)
print(a)
