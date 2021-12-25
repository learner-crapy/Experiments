import cv2
import matplotlib.pyplot as plt
import numpy as np

def add_demo(m1, m2):  # 加
    dst = cv2.add(m1, m2)
    plt.subplot(233)
    plt.title('add')
    plt.imshow(dst)


def subtract_demo(m1, m2):  # 减
    dst = cv2.subtract(m1, m2)
    plt.subplot(234)
    plt.title('sub')
    plt.imshow(dst)

def divide_demo(m1, m2):  # 除
    dst = cv2.divide(m1, m2)
    plt.subplot(235)
    plt.title('divide')
    plt.imshow(dst)

def multiply_demo(m1, m2):  # 乘
    dst = cv2.multiply(m1, m2)
    plt.subplot(236)
    plt.title('multi')
    plt.imshow(dst)
    

if __name__ == "__main__":
    src1 = cv2.imread("../Imgs/shanshui1.jpg")  # blue green red
    src2 = cv2.imread("../Imgs/shanshui2.jpg")
    plt.subplot(231)
    plt.title('original img1')
    plt.imshow(src1)

    plt.subplot(232)
    plt.title('original img2')
    plt.imshow(src2)
    add_demo(src1, src2)
    subtract_demo(src1, src2)
    divide_demo(src1, src2)
    multiply_demo(src1, src2)
    plt.show()
