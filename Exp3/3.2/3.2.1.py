import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread("../../Imgs/nature.jpg", 0)  # 灰度图像
# img2 = cv2.imread("test2.jpg")     #彩色图像
hist = cv2.calcHist([img1],  # 图像
                    [0],  # 使用的通道
                    None,  # 没有使用mask
                    [256],  # HistSize
                    [0.0, 255.0])  # 直方图柱的范围


# print(hist.shape)
# print(np.argsort(hist.reshape(256,)))

def HistGraphGray(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    histGraph = np.zeros([256, 256, 3], np.uint8)
    m = max(hist)
    hist = hist * 220 / m
    for h in range(256):
        n = int(hist[h])
        cv2.line(histGraph, (h, 255), (h, 255 - n), color)
    return histGraph


color = [255, 255, 255]
histGraph1 = HistGraphGray(img1, color)

dst = cv2.equalizeHist(img1)
dis_hist = HistGraphGray(dst, color)

plt.subplot(221)
plt.title('original img')
plt.imshow(img1, cmap='gray')

plt.subplot(222)
plt.title('original img hostGraph')
plt.imshow(histGraph1)

plt.subplot(223)
plt.title('dst')
plt.imshow(dst, cmap='gray')

plt.subplot(224)
plt.title('dst img hostGraph')
plt.imshow(dis_hist)
plt.show()

# cv2.imshow("Hist Gray", histGraph1)


# cv2.waitKey(0)
# cv2.destroyAllWindows()
