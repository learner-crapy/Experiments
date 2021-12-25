# -*- coding: utf-8 -*-
import cv2

# 读取图像
img = cv2.imread('../../Imgs/nature.jpg', 0)

# 计算x方向边缘信息
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# 计算y方向边缘信息
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
# 求绝对值
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
# x方向和y方向的边缘叠加
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

# 显示图像
cv2.imshow("origin image", img)
# cv2.imshow("x", sobelx)
# cv2.imshow("y", sobely)
cv2.imshow("xy", sobelxy)

cv2.waitKey(0)
cv2.destroyAllWindows()