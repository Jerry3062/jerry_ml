import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

x = np.random.normal(size=1000)
sns.distplot(x, bins=5, hist=False,rug=True)
plt.show()
