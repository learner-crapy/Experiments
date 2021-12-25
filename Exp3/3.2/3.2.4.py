from matplotlib import pyplot as plt
import numpy as np
from skimage import data, color


# 中文显示工具函数
def set_ch():
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']
    mpl.rcParams['axes.unicode_minus'] = False


set_ch()
D = 10
new_img = data.coffee()
new_img = color.rgb2gray(new_img)
# 傅里叶变换
f1 = np.fft.fft2(new_img)
# 使用np.fft.fftshift()函数实现平移，让直流分量输出图像的重心
f1_shift = np.fft.fftshift(f1)
# 实现理想低通滤波器
rows, cols = new_img.shape
crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
mask = np.zeros((rows, cols), dtype='uint8')  # 生成rows行，从cols列的矩阵，数据格式为uint8
# 将距离频谱中心距离小于D的低通信息部分设置为1，属于低通滤波
for i in range(rows):
    for j in range(cols):
        if np.sqrt(i * i + j * j) <= D:
            mask[crow - D:crow + D, ccol - D:ccol + D] = 1
f1_shift = f1_shift * mask
# 傅里叶逆变换
f_ishift = np.fft.ifftshift(f1_shift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))

plt.figure()
plt.subplot(121)
plt.imshow(new_img, cmap='gray')
plt.title('origina img')

plt.subplot(122)
plt.imshow(img_back, cmap='gray')
plt.title('after filter')
plt.show()