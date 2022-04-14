# 图像平滑
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


img = cv.imread("C:/Users/hmtga/Documents/open_cv/opencv_self_learning/pictures_here/dog.jpg")
# 均值滤波
blur = cv.blur(img, (5,5))   # 参数为kernel的大小
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,8), dpi=100)
axes[0].imshow(img[:,:,::-1])
axes[1].imshow(blur[:,:,::-1])
# plt.show()


# 高斯滤波
blur_1 = cv.GaussianBlur(img, (3,3), 1)   # 参数分别为卷积核大小，标准差
fig_1, axes_1 = plt.subplots(nrows=1, ncols=2, figsize=(10,8), dpi=100)
axes_1[0].imshow(img[:,:,::-1])
axes_1[1].imshow(blur_1[:,:,::-1])
# plt.show()


# 中值滤波
# 解决椒盐噪声
blur_2 = cv.medianBlur(img, 5)
fig_2, axes_2 = plt.subplots(nrows=1, ncols=2, figsize=(10,8), dpi=100)
axes_2[0].imshow(img[:,:,::-1])
axes_2[1].imshow(blur_2[:,:,::-1])
plt.show()







