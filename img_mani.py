# 形态学操作
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 连通性
# 4邻域：(x+1,y),(x-1,y),(x,y-1),(x,y+1)
# D邻域：(x+1,y+1),(x+1,y-1),(x-1,y+1),(x-1,y-1)
# 8邻域：上面两个之和
# 4连通：在4领域内
# 8连通：在8邻域内
# m连通：q在p的4领域中，或是在p的D邻域中但两者4领域交集为空

# 腐蚀和膨胀
# 针对二维图像的高亮部分而言
# 用核结构进行扫描，腐蚀是一旦有0全为0结构，膨胀是一旦有1全为1

img = cv.imread("C:/Users/hmtga/Documents/open_cv/opencv_self_learning/pictures_here/img_mani.jpg")
# 核结构
kernel = np.ones((5,5), np.uint8)
# 图像腐蚀和膨胀
img_erosion = cv.erode(img, kernel)
img_dilate = cv.dilate(img, kernel)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,8), dpi=100)
axes[0].imshow(img)
axes[1].imshow(img_erosion)
axes[2].imshow(img_dilate)
# plt.show()


# 开运算
# 先腐蚀后膨胀，消除噪点
# 闭运算
# 先膨胀后腐蚀，消除闭合物体中的孔洞

img_1 = cv.imread("C:/Users/hmtga/Documents/open_cv/opencv_self_learning/pictures_here/open.jpg")
img_2 = cv.imread("C:/Users/hmtga/Documents/open_cv/opencv_self_learning/pictures_here/close.jpg")
kernel_1 = np.ones((10,10), np.uint8)

# 进行开闭运算
cvOpen = cv.morphologyEx(img_1, cv.MORPH_OPEN, kernel_1)
cvClose = cv.morphologyEx(img_2, cv.MORPH_CLOSE, kernel_1)
fig_1, axes_1 = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
axes_1[0,0].imshow(img_1)
axes_1[0,1].imshow(cvOpen)
axes_1[1,0].imshow(img_2)
axes_1[1,1].imshow(cvClose)
# plt.show()



# 礼帽运算和黑帽运算
# 礼帽运算用来进行背景提取，黑猫运算用来分离暗一些的斑块
img_open = cv.morphologyEx(img_1, cv.MORPH_TOPHAT, kernel_1)  # 礼帽运算
img_close = cv.morphologyEx(img_2, cv.MORPH_BLACKHAT, kernel_1)  # 黑帽运算

fig_2, axes_2 = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
axes_2[0,0].imshow(img_1)
axes_2[0,1].imshow(img_open)
axes_2[1,0].imshow(img_2)
axes_2[1,1].imshow(img_close)
plt.show()







