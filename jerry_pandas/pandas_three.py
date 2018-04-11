import pandas as pd
import numpy as np

dates = pd.date_range('20180411', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)),
                  index=dates, columns=['A', 'B', 'C', 'D'])
print(df)
df.iloc[2,2] = 12
df.loc['20180411','A'] = 40
print(df)
df.B[df.A>4] = 0
print(df)
df['F'] = np.nan
print(df)