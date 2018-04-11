import pandas as pd
import numpy as np

s = pd.Series([1, 3, 4, np.nan, 44, 1])
# 生成六个连续的日期的index
dates = pd.date_range('20160101', periods=6)
print(dates)
