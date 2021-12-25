import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
    添加椒盐噪声
    :param img: 原始图片
    :param prob: 噪声比例
    :return: resultImg
    '''

def noiseSP(img, prob):
    resultImg = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random()
            if rdn < prob:
                resultImg[i][j] = 0
            elif rdn > thres:
                resultImg[i][j] = 255
            else:
                resultImg[i][j] = img[i][j]
    return resultImg

img0 = cv2.imread('../../Imgs/nature.jpg')
resultImg = noiseSP(img0, 0.05)
cv2.imwrite('../../Imgs/result_salt_paper.jpg', resultImg)





########     四个不同的滤波器    #########
img = cv2.imread('../../Imgs/result.jpg')

# 均值滤波
img_mean = cv2.blur(img, (5, 5))

# 高斯滤波
# img_Guassian = cv2.GaussianBlur(img, (5, 5), 0)

# 中值滤波
img_median = cv2.medianBlur(img, 5)

# 双边滤波
# img_bilater = cv2.bilateralFilter(img, 9, 75, 75)

# 展示不同的图片
# titles = ['srcImg', 'mean', 'Gaussian', 'median', 'bilateral']
# imgs = [img, img_mean, img_Guassian, img_median, img_bilater]

titles = ['original img', 'srcImg', 'mean', 'median']
imgs = [img0, img, img_mean, img_median]

for i in range(4):
    plt.subplot(2, 2, i + 1)  # 注意，这和matlab中类似，没有0，数组下标从1开始
    plt.imshow(imgs[i])
    plt.title(titles[i])
plt.show()
