import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("../Imgs/exp5.jpg")
k = np.ones((3, 3), np.uint8)
img = cv.morphologyEx(image, cv.MORPH_GRADIENT, k)
# cv.imshow("image", image)
# cv.imshow("morphologyEx", img)
plt.subplot(121)
plt.imshow(image, cmap="gray")
plt.title('original image')

plt.subplot(122)
plt.imshow(img, cmap='gray')
plt.title('side')

plt.show()

# cv.waitKey()
# cv.destroyAllWindows()
