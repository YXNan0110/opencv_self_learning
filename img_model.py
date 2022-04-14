# 模板匹配和霍夫变换
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 模板匹配
# 输入图像(W*H)和模板(w*h)，输出矩阵R大小为(W-w+1,H-h+1)
# res = cv.matchTemplate(img,template,method)
# method包括平方差匹配CV_TM_SQDIFF，相关匹配CV_TM_CCORR，相关系数匹配CV_TM_CCOEFF
# 匹配后还需查找max位置，平方差匹配min是最佳匹配位置

img_people = cv.imread("C:/Users/hmtga/Documents/open_cv/people.jpg")
img_person = cv.imread("C:/Users/hmtga/Documents/open_cv/person.jpg")
height, width, length = img_person.shape
# 模板匹配
res = cv.matchTemplate(img_people, img_person, cv.TM_CCORR)
# 获取最匹配位置
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
# 画匹配矩形
# 从图像中获取参数是（纵轴，横轴，层数），给图像中的点赋值或操作是（横坐标，纵坐标）
top_left = max_loc
bottom_right = (top_left[0]+width, top_left[1]+height)
cv.rectangle(img_people, top_left, bottom_right, (0,0,255), thickness=2)
plt.imshow(img_people[:,:,::-1])
# plt.show()


# 霍夫变换
# 霍夫空间的点对应笛卡尔坐标系中的线

# 霍夫线检测
# cv.HoughLines(img, rho, theta, threshold)参数分别为rho和theta的精确度，以及高于该阈值才被认为是直线
img = cv.imread("C:/Users/hmtga/Documents/open_cv/calendar.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150)# 边缘检测

lines = cv.HoughLines(edges, 0.8, np.pi /180, 150)
# lines是图像中所有被检测到的线
# 将这些线转换为笛卡尔坐标系再绘制在图片中
for line in lines:
    rho, theta = line[0]
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    x1 = int(x + 1000 * (-np.sin(theta)))
    y1 = int(y + 1000 * (np.cos(theta)))
    x2 = int(x - 1000 * (-np.sin(theta)))
    y2 = int(y - 1000 * (np.cos(theta)))
    cv.line(img, (x1, y1), (x2, y2), (0,255,0))

plt.imshow(img[:,:,::-1])
# plt.show()


# 霍夫圆检测
# circles = cv.HoughCircles(image, method, dp, minDist, param1=100, param2=100, minRadius=0,maxRadius=0 )
# method参数是CV_HOUGH_GRADIENT，dp是分辨率，dp=2时霍夫空间为输入空间的1/2，param1是Canny算子高阈值，param2是检测圆心和半径的阈值
# 返回[圆心横坐标，圆心纵坐标，圆半径]


