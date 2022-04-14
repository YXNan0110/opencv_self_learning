# 绘制几何图形
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = np.zeros((512,512,3), np.uint8)

cv.line(img, (0,0), (511,511), (0,0,255), thickness=2)
cv.circle(img, (255,255), 50, (0,255,0), thickness=1)
cv.rectangle(img, (205,205), (305,305), (255,0,0), thickness=1)
cv.putText(img, "OPENCV", (10,500), cv.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2,cv.LINE_AA)

plt.imshow(img[:,:,::-1])
plt.show()

px = img[100,100]
print(px)
red = img[100,100,2]    # 第三个参数为BGR的参数，分别为blue-0，green-1，red-2，坐标为(100,100)
print(red)

# 拆分BGR通道
b, g, r = cv.split(img)
# 通道合并
img = cv.merge ((b,g,r))

# 当图片数组为RGB格式时可以利用拆分通道转换为BGR格式
img_new = cv.imread("C:/Users/hmtga/Documents/open_cv/opencv_self_learning/pictures_here/img1.jpg")
r1, g1, b1 = cv.split(img_new)
img_resp = cv.merge((b1, g1, r1))

plt.imshow(img_resp)     # 这样显示也是正确的
plt.show()

new_img = cv.cvtColor(img_new, cv.COLOR_BGR2GRAY)

plt.imshow(new_img, cmap=plt.cm.gray)    # cmap是调整颜色，相当于调色盘
plt.show()
