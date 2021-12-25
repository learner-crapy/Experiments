import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('../../Imgs/nature.jpg')
v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)
res = np.hstack((v1, v2))
plt.subplot(1, 2, 1)
plt.title("original image")
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.title("side of canny")
plt.imshow(res, cmap="gray")
plt.show()
# cv2.imshow('res', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()