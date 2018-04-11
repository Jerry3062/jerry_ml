import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(1000)
y = np.random.rand(1000)
size = np.random.rand(1000) * 50
colors = np.random.rand(1000)
# size 点大小 colors 点颜色
plt.scatter(x, y, size, colors)
# 添加颜色栏
plt.colorbar()
plt.show()
