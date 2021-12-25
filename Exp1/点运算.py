import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("../Imgs/nature.jpg")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 获取图像的高和宽
grayImg_height = grayImg.shape[0]
grayImg_width = grayImg.shape[1]

# 创建新图像
newImg_increase = np.zeros((grayImg_height, grayImg_width), np.uint8)  # 对比度增强
newImg_decrease = np.zeros((grayImg_height, grayImg_width), np.uint8)  # 对比度减弱
# DB=DA*1.5 对比度增强
for i in range(grayImg_height):
    for j in range(grayImg_width):
        if (int(grayImg[i, j] * 1.5) > 255):
            gray = 255
        else:
            gray = int(grayImg[i, j] * 1.5)
        newImg_increase[i, j] = np.uint8(gray)

# DB=DA*0.8 对比度减弱
for i in range(grayImg_height):
    for j in range(grayImg_width):
        gray = int(grayImg[i, j]) * 0.1
        newImg_decrease[i, j] = np.uint8(gray)

# 原始图像
img0 = plt.subplot(2, 2, 1)
img0.set_title("original img")
plt.imshow(grayImg, cmap="gray")

# newImg_increase 对比度增强
img2 = plt.subplot(2, 2, 2)
img2.set_title("lighter")
plt.imshow(newImg_increase, cmap="gray")

img3 = plt.subplot(2, 2, 3)
img3.set_title("darker")
plt.imshow(newImg_decrease, cmap="gray")

# 图像求补
Img_Gray4 = 255-grayImg
plt.subplot(2, 2, 4)
plt.title('inverse')
plt.imshow(Img_Gray4, cmap="gray")
plt.show()
