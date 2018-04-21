import cv2
import numpy as np
from scipy import ndimage

img = cv2.imread('emmawatson.jpg',0)
blured = cv2.GaussianBlur(img,(11,11),0)
cv2.imshow('a',blured)
cv2.waitKey()