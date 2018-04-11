import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 50)
plt.plot(x, np.sin(x), 'r-x', label='Sin')
plt.plot(x, np.cos(x), 'g-', label='Cos')
plt.legend()  # 展示图例
plt.xlabel('Rads')
plt.ylabel('Amplitude')
plt.title('Sin and Cos Wwaves')
plt.show()
