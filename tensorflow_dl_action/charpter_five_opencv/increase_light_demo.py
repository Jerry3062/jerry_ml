import cv2
import numpy as np
from scipy import ndimage

# 通过增加亮度卷积核处理
kernel133 = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]])
img = cv2.imread('emmawatson.jpg',0)
light_img = ndimage.convolve(img, kernel133)
cv2.imshow('1', light_img)
cv2.waitKey()
