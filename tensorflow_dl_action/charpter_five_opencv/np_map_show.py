import numpy as np
import cv2

img = np.mat(np.ones((300, 300)) * 255, dtype=np.uint8)
img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
cv2.imshow("haha", img)
cv2.waitKey(0)
print(img)

# 对于图片来说，可以通过Python自带的方法，将其转化成标准的一维Python bytearray格式
imageByteArray = bytearray(img)
print(len(imageByteArray))
# 同样。bytearray可以通过矩阵重构的方法还原为原本的图片矩阵
imageBGR = np.array(imageByteArray).reshape(300,300,3)
print(imageBGR.shape)