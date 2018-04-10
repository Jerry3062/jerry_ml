from sklearn import preprocessing
import jerry_numpy as np

a = np.array([[10, 2.7, 3.6],
              [-100, 5, -2],
              [120, 20, 40]], dtype=np.float32)
print(a)
print(preprocessing.scale(a))

print(preprocessing.minmax_scale(a,feature_range=(0,1)))