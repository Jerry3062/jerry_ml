import pandas as pd
import numpy as np

dates = pd.date_range('20180411', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates
                  , columns=['A', 'B', 'C', 'D'])
# print(df)
# print(df.A)
# print(df[1:3])
# select by label:loc
# print(df.loc['20180411'])
# print(df.loc[:, ['A', 'B']])
# print(df.loc['20180411', ['A', 'B']])

# select by position:iloc
# print(df.iloc[1])
# print(df.iloc[[1,3,5],1:3])
#
#mixed selection:ix
# print(df.ix[:3,['A','C']])

print(df[df.A>8])
