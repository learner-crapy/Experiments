from math import fabs, sin, radians, cos
import matplotlib.pyplot as plt
# fabs()用于求高精度的绝对值
# radians()将角度转换为弧度
import cv2
import numpy as np


def rotate_img(img, degrees):
    '''
    旋转图片
    :param img: 原始图片
    :param degrees: 旋转角度
    :return:
    '''
    height, width = img.shape[:2]
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degrees))) + height * fabs(cos(radians(degrees))))
    widthNew = int(height * fabs(sin(radians(degrees))) + width * fabs(cos(radians(degrees))))

    # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
    # 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    resultImg = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    return resultImg


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


def resize_proportion(img, h_zoom, w_zoom):
    '''
    缩放
    :param img: 原始图片
    :param h_zoom: 纵向比例
    :param w_zoom: 横向比例
    :return: 
    '''
    height, width = img.shape[:2]
    resultImg = cv2.resize(img, (int(width * w_zoom), int(height * h_zoom)))
    return resultImg


def cropped_center(img, x1, y1, x2, y2):
    '''
    裁剪
    :param img: 原始图片
    :param x1: 左边界
    :param y1: 上边界
    :param x2: 右边界
    :param y2: 下边界
    :return:
    '''
    resultImg = img[y1:y2, x1:x2]
    return resultImg
def jingxiang(imgpath):
    img = cv2.imread(imgpath, 1)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    deep = imgInfo[2]
    dst = np.zeros([height * 2, width, deep], np.uint8)
    for i in range(height):
        for j in range(width):
            dst[i, j] = img[i, j]
            dst[height * 2 - i - 1, j] = img[i, j]
    for i in range(width):
        dst[height, i] = (0, 0, 255)
    return dst


if __name__ == '__main__':
    img = cv2.imread('../Imgs/nature.jpg')

    plt.subplot(2, 3, 1)
    plt.title('original img')
    plt.imshow(img)

    plt.subplot(2, 3, 2)
    plt.title('rotate')
    resultImg = rotate_img(img, 45)
    cv2.imwrite('../Imgs/nature_rotate.jpg', resultImg)
    plt.imshow(resultImg)

    resultImg = pan_img(img, -50, -50)
    plt.subplot(2, 3, 3)
    plt.title('pingyi')
    plt.imshow(resultImg)

    resultImg = resize_proportion(img, 1, 0.5)
    plt.subplot(2, 3, 4)
    plt.title('resize')
    plt.imshow(resultImg)

    resultImg = cropped_center(img, 20, 20, 400, 400)
    plt.subplot(2, 3, 5)
    plt.title('crop')
    plt.imshow(resultImg)

    resultImg = jingxiang('../Imgs/nature.jpg')
    plt.subplot(2, 3, 6)
    plt.title('jingxiang')
    plt.imshow(resultImg)

    plt.show()


