# the multiplicative noise confirm to the rayleigh noise
# the distribution function as the following
# f(x, sigma) = (x/(sigma)**2)*exp(-x**2/2*sigma**2), x>0
import math
import numpy as np
import random
import cv2
from matplotlib import pyplot as plt


def Multiplicative_Noise(image, sigma=0.5):
    image = image/255.0
    noise = np.zeros(image.shape)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            x = random.randint(0, 10)
            noise[i][j] = (x / (sigma) ** 2) * math.exp(-x ** 2 / (2 * sigma ** 2))
    out = image + noise
    out = np.clip(out, 0, 1.0)
    out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out


# Read image
img = cv2.imread("../../Imgs/nature.jpg")
# 添加高斯噪声，均值为0，方差为0.001
out2 = Multiplicative_Noise(img)
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
plt.title('multiplicative noise')
plt.imshow(out2)

plt.subplot(223)
plt.title('original mean')
plt.imshow(img_mean)

plt.subplot(224)
plt.title('original median')
plt.imshow(img_median)
plt.show()
