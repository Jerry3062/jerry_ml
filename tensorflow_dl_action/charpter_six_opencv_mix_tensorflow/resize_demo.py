import cv2

img = cv2.imread('1.jpg')
cv2.imshow('a',img)
img2 = cv2.resize(img,(200,200))
cv2.imshow('b',img2)
cv2.waitKey()