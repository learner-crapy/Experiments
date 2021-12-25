import cv2
import random
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('../Imgs/nature.jpg')
# 图像灰度化
Img_Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 图像二值化
def threshold_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 要二值化图像，要先进行灰度化处理
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary
binary = threshold_demo(image)

plt.subplot(131)
plt.title("original img")
plt.imshow(image)

plt.subplot(132)
plt.title('gray')
plt.imshow(Img_Gray, cmap='gray')

plt.subplot(133)
plt.title('binary')
plt.imshow(binary, cmap='gray')

plt.show()
