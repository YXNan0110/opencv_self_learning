# 图像特征
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Harris角点检测
# 角点处灰度变化明显
# 通过角点响应值R来判断，正为角点，负为边界，绝对值小为平坦区域
# dst=cv.cornerHarris(src, blockSize, ksize, k)
# 图像必须为float32的数据类型，blockSize是邻域大小，ksize是Sobel核大小，k是自由参数(0.04,0.06)


img = cv.imread("C:/Users/hmtga/Documents/open_cv/table.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

res = cv.cornerHarris(gray, 2, 3, 0.04)
# 绘制角点
img[res>0.001*res.max()] = [0,0,255]

plt.imshow(img[:,:,::-1])
plt.show()

# 估计大家最有疑问的应该是：
# 　　　　img[dst>0.01*dst.max()]=[0,0,255]这段代码是什么意思吧　    dst>0.01*dst.max()这么多返回是满足条件的dst索引值　　根据索引值来设置这个点的颜色
# 　　　　这里是设定一个阈值　当大于这个阈值分数的都可以判定为角点　　（好像说跟没说一样是吧）hhhh
# 　　　　在看上面我讲的　这里的dst其实就是一个个角度分数R组成的　　　当 λ 1 和 λ 2 都很大，并且 λ 1 ～λ 2 中的时，R 也很大，（λ 1 和 λ 2 中的最小值都大于阈值）说明这个区域是角点。
# 　　　　那么这里为什么要大于０．０１×ｄｓｔ.max()呢　注意了这里Ｒ是一个很大的值　我们选取里面最大的Ｒ　然后　只要dst里面的值大于百分之一的Ｒ的最大值　　那么此时这个dst的Ｒ值也是很大的　可以判定他为角点
# 　　　　也不一定要０．０１　　　可以根据图像自己选取不过如果太小的话　可能会多圈出几个不同的角点


# Shi-Tomasi 角点检测
# corners = cv2.goodFeaturesToTrack ( image, maxcorners, qualityLevel, minDistance )
# quality角点质量水平，在0-1之间，minDistance角点之间最小距离，返回Corners搜索到的角点

img_cal = cv.imread("C:/Users/hmtga/Documents/open_cv/tv.jpg")
cal_gray = cv.cvtColor(img_cal, cv.COLOR_BGR2GRAY)

Corners = cv.goodFeaturesToTrack(cal_gray, 1000, 0.1, 5)
# 绘制角点
for i in Corners:
    x, y = i.ravel()   # 将数组拉成一维数组
    cv.circle(img_cal, (int(x),int(y)), 1, [0,0,255], -1)

plt.imshow(img_cal[:,:,::-1])
plt.show()


# SIFT原理
# sift = cv.xfeatures2d.SIFT_create()
# kp,des = sift.detectAndCompute(gray,None)
# kp是关键点信息，包括位置、尺度、方向信息；des是关键点描述符，每个关键点对应128个梯度信息的特征向量
# cv.drawKeypoints(image, keypoints, outputimage, color, flags)
# flags是标识设置，DRAW_MATCHES_FLAGS_DEFAULT是输出图像矩阵且只绘制中间点，DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG是不输出矩阵，而是绘制匹配对

img_3 = img_cal
sift = cv.xfeatures2d.SIFT_create()

kp, des = sift.detectAndCompute(cal_gray, None)
# 绘制检测结果
resu = cv.drawKeypoints(img_3, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(resu[:,:,::-1])
plt.show()


# FAST算法
# fast = cv.FastFeatureDetector_create(threshold, nonmaxSuppression)   参数为阈值和是否有极大值抑制（默认有）
# 返回创建的FastFeatureDetector对象
# kp = fast.detect(grayImg, None)    返回关键点参数：位置、尺度、方向
# cv.drawKeypoints(image, keypoints, outputimage, color, flags)

img_2 = img_cal
fast = cv.FastFeatureDetector_create(threshold=30)
kp_1 = fast.detect(cal_gray, None)

img_new = cv.drawKeypoints(img_2, kp_1, None, [0,0,255])
# 输出默认参数
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp_1)) )

plt.imshow(img_new[:,:,::-1])
plt.show()


# ORB算法
# orb = cv.xfeatures2d.orb_create(nfeatures)      特征点的最大数量
# kp,des = orb.detectAndCompute(gray,None)
# cv.drawKeypoints(image, keypoints, outputimage, color, flags)

img_1 = img_cal
orb = cv.ORB_create(nfeatures=500)
kp_2, des_2 = orb.detectAndCompute(cal_gray, None)
out_img = cv.drawKeypoints(img_1, kp_2, None, [0,0,255], flags=0)
plt.imshow(out_img[:,:,::-1])
plt.show()









