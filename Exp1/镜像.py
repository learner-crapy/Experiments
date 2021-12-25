import cv2
import matplotlib.pyplot as plt
import numpy as np

def jingxiang(imgpath):
    img = cv2.imread(imgpath, 1)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    deep = imgInfo[2]
    dst = np.zeros([height * 2, width, deep], np.uint8)
    for i in range(height):
        for j in range(width):
            dst[i, j] = img[i, j]
            dst[height * 2 - i - 1, j] = img[i, j]
    for i in range(width):
        dst[height, i] = (0, 0, 255)
    return dst

