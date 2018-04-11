import pandas as pd
import numpy as np

dates = pd.date_range('20180411', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)),
                  index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan
# print(df)
# 丢掉行 how={any,all} any 有nan就丢 all 所有nan才丢
# print(df.dropna(axis=0,how='any'))#
# 填充nan的数据
# print(df.fillna(value=0))
#至少有一个是丢失了的
print(np.any(df.isnull()) == True)
