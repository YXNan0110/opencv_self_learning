# 几何变换
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("C:/Users/hmtga/Documents/open_cv/face_detection/img1.jpg")

# 图片尺寸
rows, cols = img.shape[:2]    # shape输出三个参数，分别为高度，宽度，通道数
img_1 = cv.resize(img,(2*cols,2*rows), interpolation=cv.INTER_CUBIC)   # 插值方法为双三次插值法
# INTER_LINEAR为双线性插值法，INTER_NEAREST为最邻近插值，INTER_AREA为（默认）像素区域重采样

# 相对尺寸缩放
img_2 = cv.resize(img, None, fx=0.5, fy=0.5)  # 令dsize参数为None，fx，fy为比例因子
# dsize参数是（宽度，高度）的形式，宽度是列数！高度是行数！

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,8), dpi=100)
# 放大
axes[0].imshow(img_1[:,:,::-1])
# 原图
axes[1].imshow(img[:,:,::-1])
# 缩小
axes[2].imshow(img_2[:,:,::-1])

plt.show()



# 平移矩阵
# （2*3）的移动矩阵，[1,0,x][0,1,y]格式为float32
M = np.float32([[1,0,100],[0,1,50]])
move_img = cv.warpAffine(img, M, (cols,rows))   # dsize参数用来设置图片大小
plt.imshow(move_img[:,:,::-1])
plt.show()



# 旋转矩阵
M1 = cv.getRotationMatrix2D((cols/2,rows/2), 90, 1)   # 参数设置分别为旋转中心，旋转角度，缩放比例
# 旋转
rotation_img = cv.warpAffine(img, M1, (cols,rows))
plt.imshow(rotation_img[:,:,::-1])
plt.show()


# 仿射变换
# 仿射变换就相当于把一个面进行扭曲，每个面需要确立三个点进行一一映射
# 变换矩阵
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[100,100],[200,50],[100,250]])
M2 = cv.getAffineTransform(pts1, pts2)

new_img = cv.warpAffine(img, M2, (cols,rows))
plt.imshow(new_img[:,:,::-1])
plt.show()


# 透射变换
# 由光源发出一束光线，通过投影面投射在新的平面上，形成新的图像，投射矩阵需要找四个面上的点
# 投射矩阵
pts3 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts4 = np.float32([[100,145],[300,100],[80,290],[310,300]])
M3 = cv.getPerspectiveTransform(pts3, pts4)

per_img = cv.warpPerspective(img, M3, (cols,rows))
plt.imshow(per_img[:,:,::-1])
plt.show()

# 上采样
# 图像变大，分辨率增加
# up_img = cv.pyrUp(img)
# 下采样
# 图像缩小，分辨率降低
# down_img = cv.pyrDown(img)



