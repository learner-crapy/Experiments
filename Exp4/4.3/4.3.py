import matplotlib.pyplot as plt
import numpy as np
import cv2


def watershed(imgpath):
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret0, thresh0 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh0, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 确定前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret1, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 查找未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记标签
    ret2, markers1 = cv2.connectedComponents(sure_fg)
    markers = markers1 + 1
    markers[unknown == 255] = 0

    markers3 = cv2.watershed(img, markers)
    img[markers3 == -1] = [0, 255, 0]
    return thresh0, sure_bg, sure_fg, img


if __name__ == '__main__':
    imgpath = '../../Imgs/yingbi.jpg'
    img0 = cv2.imread(imgpath)
    thresh0, sure_bg, sure_fg, img = watershed(imgpath)
    imgs = [img0, thresh0, sure_bg, sure_bg, img]
    # cv2.imshow('thresh0',thresh0)
    # cv2.imshow('sure_bg', sure_bg)
    # cv2.imshow('sure_fg', sure_fg)
    # cv2.imshow('result_img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    titles = ['orignal img', 'thread', 'bg', 'fg', 'result']
    # plt.figure(5, (5, 5))
    for i in range(0, 5):
        plt.subplot(1, 5, i + 1)
        plt.title(titles[i])
        plt.imshow(imgs[i], cmap='gray')
    plt.show()
