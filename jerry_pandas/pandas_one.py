import pandas as pd
import numpy as np

s = pd.Series([1, 3, 4, np.nan, 44, 1])
# 生成六个连续的日期的index
dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
# print(df)
df2 = pd.DataFrame({'A': 1, 'B': [2, 3]})
# print(df2)
# print(df2.dtypes)
print(df2.columns)
print(df2['B'])
print(df2.values)
print(df2.describe())
# axis=1对列名 ascending=False倒排序
print(df2.sort_index(axis=1, ascending=False))
# 按照A列排序，升序
print(df2.sort_index(by='A', ascending=True))
