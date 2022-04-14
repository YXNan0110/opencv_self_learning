# 边缘检测
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 基于搜索来检测边界：寻找一阶导数最大值，如Sobel算子和Scharr算子
# 基于零穿越来检测边界：寻找二阶导数零点，如Laplacian算子

# Sobel算子
# 先将图像进行x，y方向分别卷积，因为溢出问题，所以要用16位的CV_16S，再将数据进行格式转换为8位
img = cv.imread("C:/Users/hmtga/Documents/open_cv/horse.jpg", 0)
# Sobel卷积
# Sobel_x_or_y = cv2.Sobel(src, ddepth, dx, dy, dst, ksize, scale, delta, borderType)
x = cv.Sobel(img, cv.CV_16S, 1, 0)    # 对谁卷积谁就是1，剩下的就是0
y = cv.Sobel(img, cv.CV_16S, 0, 1)
# 数据转换
Scale_abs_x = cv.convertScaleAbs(x)
Scale_abs_y = cv.convertScaleAbs(y)
# 结果合成
res = cv.addWeighted(Scale_abs_x, 0.5, Scale_abs_y, 0.5, 0)
plt.figure(figsize=(10,8), dpi=100)
plt.subplot(121), plt.imshow(img, cmap=plt.cm.gray)
plt.subplot(122), plt.imshow(res, cmap=plt.cm.gray)
# plt.show()


# Scharr边缘检测更准确
# 将Sobel中的ksize参数设置为-1
m = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=-1)
n = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=-1)
Scale_abs_m = cv.convertScaleAbs(m)
Scale_abs_n = cv.convertScaleAbs(n)
res_1 = cv.addWeighted(Scale_abs_m, 0.5, Scale_abs_n, 0.5, 0)
plt.figure(figsize=(10,8), dpi=100)
plt.subplot(121), plt.imshow(img, cmap=plt.cm.gray)
plt.subplot(122), plt.imshow(res_1, cmap=plt.cm.gray)
# plt.show()


# Laplacian算子
# laplacian = cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])
res_2 = cv.Laplacian(img, cv.CV_16S)
Scale_abs = cv.convertScaleAbs(res_2)
plt.figure(figsize=(10,8), dpi=100)
plt.subplot(121), plt.imshow(img, cmap=plt.cm.gray)
plt.subplot(122), plt.imshow(res_2, cmap=plt.cm.gray)
# plt.show()


# Canny边缘检测
# 最优边缘检测算法
# 噪声去除，计算图像梯度，非极大值抑制，滞后阈值：minVal, maxVal
# canny = cv2.Canny(image, threshold1, threshold2)
lowThreshold = 0
max_Threshold = 100
canny = cv.Canny(img, lowThreshold, max_Threshold)
plt.figure(figsize=(10,8), dpi=100)
plt.subplot(121), plt.imshow(img, cmap=plt.cm.gray)
plt.subplot(122), plt.imshow(canny, cmap=plt.cm.gray)
plt.show()




