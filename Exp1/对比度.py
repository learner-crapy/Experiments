import cv2
import random
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('../Imgs/nature.jpg')
# 图像灰度化
Img_Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(2, 2, 1)
plt.title('original gray')
plt.imshow(Img_Gray, cmap="gray")

# 增大对比度
Img_Gray2 = Img_Gray*5
plt.subplot(2, 2, 2)
plt.title('lighter')
plt.imshow(Img_Gray2, cmap="gray")
# plt.show()

# 减小对比度
Img_Gray3 = Img_Gray*0.0001
plt.subplot(2, 2, 3)
plt.title('darker')
plt.imshow(Img_Gray3, cmap="gray")

# 图像求补
Img_Gray4 = 255-Img_Gray
plt.subplot(2, 2, 4)
plt.title('inverse')
plt.imshow(Img_Gray4, cmap="gray")
plt.show()