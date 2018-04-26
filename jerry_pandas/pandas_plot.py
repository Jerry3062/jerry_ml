import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data = pd.Series(np.random.randn(1000),index=np.arange(1000))
# data = data.cumsum()
# data.plot()
# plt.show()
#
# data = pd.DataFrame(np.random.randn(1000,4),
#                     index=np.arange(1000),
#                     columns=list("ABCD"))
# data = data.cumsum()
# data.plot()
# plt.show()

# plot methods
# bar hist box kde area scatter hexbin pie

a = np.array([[1, 2], [3, 4]])
sum0 = np.sum(a, axis=0)
sum1 = np.sum(a, axis=1)
print(sum0)
print(sum1)
