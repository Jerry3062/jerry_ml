import pandas as pd
import numpy as np

# concatenating
df0 = pd.DataFrame(np.ones((3, 4)) * 0, columns=['A', 'B', 'C', 'D'])
df1 = pd.DataFrame(np.ones((3, 4)) * 1, columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame(np.ones((3, 4)) * 2, columns=['A', 'B', 'C', 'D'])
# ignore_index 忽略原index
print(pd.concat([df0, df1, df2], axis=0, ignore_index=True))
