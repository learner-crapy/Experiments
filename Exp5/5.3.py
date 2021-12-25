# coding:utf-8
import cv2
import numpy as np

im = cv2.imread('../Imgs/renwu.png', cv2.IMREAD_GRAYSCALE)
thresh, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('binary.png', im)  # 控制背景为黑色
# cv2.imwrite('binary.png',im)
cv2.waitKey(0)

arr = np.array(im)
mat = []  # HMT模板矩阵
mat.append(np.array([[-1, -1, -1], [0, 1, 0], [1, 1, 1]]))
mat.append(np.array([[0, -1, -1], [1, 1, -1], [1, 1, 0]]))
mat.append(np.array([[1, 0, -1], [1, 1, -1], [1, 0, -1]]))
mat.append(np.array([[-1, -1, -1], [0, 1, 0], [1, 1, 1]]))
mat.append(np.array([[1, 1, 0], [1, 1, -1], [0, -1, -1]]))
mat.append(np.array([[1, 1, 1], [0, 1, 0], [-1, -1, -1]]))
mat.append(np.array([[-1, 0, -1], [-1, 1, 1], [-1, 0, 1]]))
mat.append(np.array([[-1, -1, 0], [-1, 1, 1], [0, 1, 1]]))

height, width = arr.shape

count = 0

while True:  # 迭代直至无变化
    before = arr.copy()
    for m in mat:  # 使用八个模板进行变换
        mark = []
        for i in range(height - 2):  # 对每个非边界点进行测试
            for j in range(width - 2):
                reg = True
                for im in range(3):
                    for jm in range(3):
                        print(1)
                        if not arr[i + 1][j + 1] == 255:
                            continue
                        if m[im][jm] == 1 and arr[i + im][j + jm] == 0:
                            reg = False
                        if m[im][jm] == -1 and arr[i + im][j + jm] == 255:
                            reg = False
                if reg:  # 找到标记，删除
                    mark.append((i + 1, j + 1))
        for it in mark:
            x, y = it
            arr[x][y] = 0
    if (before == arr).all():
        print("break")
        break
    # if (err >= 1000000):
    #     break
    count = count + 1
    print("count:" + str(count))
    cv2.imshow('bthin', arr)
    cv2.waitKey(0)

if (True):
    cv2.imshow('thin.png', arr)
    cv2.imwrite('thin.png', arr)
    cv2.waitKey(0)

cv2.destroyAllWindows()
