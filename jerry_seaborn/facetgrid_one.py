import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='ticks')

tips = sns.load_dataset('tips')
# print(tips)
g = sns.FacetGrid(tips, col='sex', hue='smoker')
g.map(plt.scatter, 'total_bill','tip')
g.add_legend()
plt.show()
