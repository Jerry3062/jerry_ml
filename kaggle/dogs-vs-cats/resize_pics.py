import cv2
import os


def rebuild(dir):
    for root, dirts, files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                image = cv2.imread(filepath)
                dim = (227, 227)
                resized = cv2.resize(image, dim)
                path = 'F:/dataset/dogs-vs-cats-resized/' + file
                cv2.imwrite(path, resized)
            except:
                print(filepath, '===>this image is not good')
                os.remove(filepath)
        cv2.waitKey(0)


rebuild('F:/dataset/dogs-vs-cats/train')
