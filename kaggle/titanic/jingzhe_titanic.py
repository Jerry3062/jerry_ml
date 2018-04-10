# 知乎 惊蛰 的专栏分享
import numpy as np
import seaborn
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 50)
print(x)
plt.subplot(2, 1, 1)
plt.plot(x, np.sin(x), 'r', x, np.sin(2 * x))
plt.show()
plt.subplot(2, 1, 1)
plt.plot(x, np.cos(x), 'g')
plt.show()
