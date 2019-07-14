import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('src.jpg')
rows, cols = img.shape[:2]

# 原图中卡片的四个角点
pts1 = np.float32([[64, 74], [161, 32], [154, 222], [271, 148]])
# 变换后分别在左上、右上、左下、右下四个点
pts2 = np.float32([[0, 0], [160, 0], [0, 178], [160, 178]])

# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)
# 进行透视变换
dst = cv2.warpPerspective(img, M, (320, 178))

plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('input')
plt.subplot(122), plt.imshow(dst[:, :, ::-1]), plt.title('output')
plt.show()