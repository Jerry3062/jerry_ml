
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

'''Seaborn有五个预设好的主题：darkgrid, whitegrid, dark, white,和ticks。
它们各自适用于不同的应用和个人喜好。缺省的主题是darkgrid。如上文提到的，
网格让图表的布局成为了查找大量信息的表格，并且白线灰底让网络不会
影响代表数据的线的显示。尽管whitegrid主题非常简洁，但是它更适用于
数据元素较大的布局。
sns.set_style("whitegrid")
'''
x = np.random.normal(size=100)
sns.set_style('whitegrid')
sns.distplot(x)
plt.show()
#ceshi
