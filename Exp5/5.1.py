import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../Imgs/fingerprint.jpg')
img1 = cv2.imread('../Imgs/fingerprint.jpg')

kernel = np.ones((5, 5), np.uint8)  # 卷积核
kernel2 = np.ones((10, 10), np.uint8)  # 卷积核

# erosion = cv2.erode(img, kernel, iterations=1)  # 腐蚀
# dilation = cv2.dilate(img, kernel, iterations=1)  # 膨胀
opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)  # 开运算
# 先进性腐蚀再进行膨胀就叫做开运算,它被用 来去除噪声
closing = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)  # 闭运算
# 先膨胀再腐蚀。它经常被用来填充前景物体中的小洞，或者前景物体上的小黑点
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)  # 形态学梯度
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel2)  # 礼帽
# 原始图像与进行开运算之后得到的图像的差
# blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel2)  # 黑帽
# 进行闭运算之后得到的图像与原始图像的差

plt.subplot(131), plt.imshow(img), plt.title('Original')
# plt.subplot(242), plt.imshow(erosion), plt.title('Erosion')
# plt.subplot(243), plt.imshow(dilation), plt.title('Dilation')
plt.subplot(132), plt.imshow(opening), plt.title('Opening')
plt.subplot(133), plt.imshow(closing), plt.title('Closing')
# plt.subplot(246), plt.imshow(gradient), plt.title('Gradient')
# plt.subplot(247), plt.imshow(tophat), plt.title('Tophat')
# plt.subplot(248), plt.imshow(blackhat), plt.title('Blackhat')

plt.show()