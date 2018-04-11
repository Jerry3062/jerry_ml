import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
iris = sns.load_dataset('iris')
sns.pairplot(iris)
plt.show()