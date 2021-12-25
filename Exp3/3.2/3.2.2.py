import numpy as np
import cv2
import matplotlib.pyplot as plt


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


# plt.figure(figsize=(5, 5))
img = plt.imread('../../Imgs/nature.jpg')
plt.subplot(221), plt.imshow(img), plt.title('picture')

# 根据公式转成灰度图
img = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
# 进行傅立叶变换
fft2 = np.fft.fft2(img)
# 对傅立叶变换的结果进行对数变换，并显示效果
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(222), plt.imshow(log_fft2, 'gray'), plt.title('log_fft2')



img = plt.imread('../../Imgs/nature.jpg')

img1 = pan_img(img, -50, -50)
plt.subplot(223), plt.imshow(img1), plt.title('pan_picture')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# 进行傅立叶变换
fft2 = np.fft.fft2(img1)
# 对傅立叶变换的结果进行对数变换，并显示效果
log_fft2 = np.log(1 + np.abs(fft2))
plt.subplot(224), plt.imshow(log_fft2, 'gray'), plt.title('pan_log_fft2')

plt.show()