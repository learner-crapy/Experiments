import numpy as np
import random
import cv2
from matplotlib import pyplot as plt


def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out


# Read image
img = cv2.imread("../../Imgs/nature.jpg")
# 添加高斯噪声，均值为0，方差为0.001
out2 = gasuss_noise(img, mean=0, var=0.001)
cv2.imwrite('../../Imgs/nature_gaussian.jpg', out2)

# 均值滤波
img_mean = cv2.blur(out2, (5, 5))

# 高斯滤波
# img_Guassian = cv2.GaussianBlur(img, (5, 5), 0)

# 中值滤波
img_median = cv2.medianBlur(out2, 5)

plt.subplot(221)
plt.title('original img')
plt.imshow(img)

plt.subplot(222)
plt.title('gaussian noise')
plt.imshow(out2)

plt.subplot(223)
plt.title('original mean')
plt.imshow(img_mean)

plt.subplot(224)
plt.title('original median')
plt.imshow(img_median)
plt.show()
