import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(1000000)
#直方图 10代表多少个直方图柱
plt.hist(x, 10)
plt.show()
