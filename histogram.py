# 灰度直方图
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 术语
# dims：维度，需要统计的特征数目
# bins：组距
# range：统计特征取值范围

# 直方图绘制
img = cv.imread("C:/Users/hmtga/Documents/open_cv/dog.jpg", 0)
histr = cv.calcHist([img], [0], None, [256], [0,256])   # 图像要加中括号，参数分别为通道，[0]or[0][1][2]，mask掩膜图像，bin的数目，像素值范围
plt.figure(figsize=(10,6),dpi=100)
plt.plot(histr)
plt.grid()
# plt.show()


# 掩膜
# 1值被处理，0值被屏蔽
# 需要先创建蒙版，然后再覆盖在原图上
# 创建蒙版
mask = np.zeros(img.shape[:2], np.uint8)
mask[0:175, 175:450] = 1   # 先是列，后是行
# 掩膜
mask_img = cv.bitwise_and(img, img, mask=mask)   # 图像求并集
mask_histr = cv.calcHist([img], [0], mask, [256], [0,256])

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
axes[0,0].imshow(img, cmap=plt.cm.gray)
axes[0,1].imshow(mask, cmap=plt.cm.gray)
axes[1,0].imshow(mask_img, cmap=plt.cm.gray)
axes[1,1].plot(mask_histr)

# plt.show()

img_new = cv.imread("C:/Users/hmtga/Documents/open_cv/histr.jpg", 0)
# 直方图均衡化
# 提高对比度
dst = cv.equalizeHist(img_new)
fig_1, axes_1 = plt.subplots(nrows=1, ncols=2, figsize=(10,8))
axes_1[0].imshow(img_new, cmap=plt.cm.gray)
axes_1[1].imshow(dst, cmap=plt.cm.gray)

# plt.show()


# 自适应直方图均衡化
# 对比度设限，解决亮处太亮和暗处太暗的问题
clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8,8))   # 参数分别是对比度限制和分块大小
# 应用于图像
self_img = clahe.apply(img_new)

fig_2, axes_2 = plt.subplots(nrows=1, ncols=2, figsize=(10,8))
axes_2[0].imshow(img_new, cmap=plt.cm.gray)
axes_2[1].imshow(self_img, cmap=plt.cm.gray)

plt.show()


