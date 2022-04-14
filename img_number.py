# 图像算术操作
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_1 = cv.imread("C:/Users/hmtga/Documents/open_cv/opencv_self_learning/pictures_here/pic_1.jpg")
img_2 = cv.imread("C:/Users/hmtga/Documents/open_cv/opencv_self_learning/pictures_here/pic_2.jpg")

img_1 = cv.resize(img_1, (256,256))
img_2 = cv.resize(img_2, (256,256))

# 图像相加
img = cv.add(img_1, img_2)

plt.imshow(img[:,:,::-1])
plt.show()

# 图像混合
img_3 = cv.addWeighted(img_1, 0.7, img_2, 0.3, 0) # alpha,beta,gama为三个参数的意义

plt.imshow(img_3[:,:,::-1])
plt.show()






