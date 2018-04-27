import numpy as np
import torch
from torch.utils.data import DataLoader

a = np.arange(6).reshape((2,3))
print(a)
b = torch.from_numpy(a)
print(b)
c = b.numpy()
print(c)
