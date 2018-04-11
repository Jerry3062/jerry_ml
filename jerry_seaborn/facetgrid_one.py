import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='ticks')

tips = sns.load_dataset('tips')
# print(tips)
tips.to_csv("tips.csv")