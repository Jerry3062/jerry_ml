import cv2
import numpy as np
import os

# randomByteArray = bytearray(os.urandom(120000))
# flatNumpyArray = np.array(randomByteArray).reshape(300, 400)
# print(randomByteArray)
# cv2.imshow('a',flatNumpyArray)
# cv2.waitKey(0)

img = cv2.imread('emmawatson.jpg',0)
print(type(img))
cv2.imshow('a',img)
cv2.waitKey(0)