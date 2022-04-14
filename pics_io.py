import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("C:/Users/hmtga/Documents/open_cv/face_detection/img1.jpg")

# cv.imshow("new_window", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

plt.imshow(img[:,:,::-1])        # 用matplotlib显示图像
plt.show()




