from math import fabs, sin, radians, cos
import matplotlib.pyplot as plt
# fabs()用于求高精度的绝对值
# radians()将角度转换为弧度
import cv2
import numpy as np


def pan_img(img, x, y):
    '''
    图片平移
    :param img: 原始图片
    :param x: 横向移动像素值
    :param y: 纵向移动像素值
    :return:
    '''
    M = np.float32([[1, 0, x], [0, 1, y]])
    resultImg = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return resultImg


def rotate_img(img, degrees):
    '''
    旋转图片
    :param img: 原始图片
    :param degrees: 旋转角度
    :return:
    '''
    height, width = img.shape[:2]
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degrees))) + height * fabs(cos(radians(degrees))))
    widthNew = int(height * fabs(sin(radians(degrees))) + width * fabs(cos(radians(degrees))))

    # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
    # 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    resultImg = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    return resultImg

img = plt.imread('../Imgs/nature.jpg')
plt.figure(6, (10, 10))
plt.subplot(321), plt.imshow(img), plt.title('picture')

# 根据公式转成灰度图
img = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
# 进行傅立叶变换
fft2 = np.fft.fft2(img)
# 对傅立叶变换的结果进行对数变换，并显示效果
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(322), plt.imshow(log_fft2, 'gray'), plt.title('log_fft2')



img = plt.imread('../Imgs/nature.jpg')

img1 = pan_img(img, -50, -50)
plt.subplot(323), plt.imshow(img1), plt.title('pan_picture')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# 进行傅立叶变换
fft2 = np.fft.fft2(img1)
# 对傅立叶变换的结果进行对数变换，并显示效果
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(324), plt.imshow(log_fft2, 'gray'), plt.title('pan_log_fft2')

img1 = rotate_img(img, 45)
plt.subplot(325), plt.imshow(img1), plt.title('rotate_picture')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# 进行傅立叶变换
fft2 = np.fft.fft2(img1)
# 对傅立叶变换的结果进行对数变换，并显示效果
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(326), plt.imshow(log_fft2, 'gray'), plt.title('rotate_log_fft2')



plt.show()