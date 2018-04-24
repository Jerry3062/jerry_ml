import os
import cv2


def resize_and_save(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            image = cv2.imread(filepath)
            resized = cv2.resize(image, (227, 227))
            path = 'F:/dataset/dogs-vs-cats-resized/train/' + file
            cv2.imwrite(path, resized)


resize_and_save('F:/dataset/dogs-vs-cats/train')
