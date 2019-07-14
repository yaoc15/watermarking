import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
#完成51W翻转的一个字效果

def find_border_point(image,image_color):
  temp = image_color.copy()
  for j in range(1,image.shape[1]-1):
    for i in range(1,image.shape[0]-1):
      y = np.zeros(10,dtype=int)
      y[1] = image[i-1][j-1]
      y[2] = image[i][j-1]
      y[3] = image[i+1][j-1]
      y[4] = image[i+1][j]
      y[5] = image[i+1][j+1]
      y[6] = image[i][j+1]
      y[7] = image[i-1][j+1]
      y[8] = image[i-1][j]
      y[9] = y[1]
      f = 0
      for k in range(1,9):
        f += abs(y[k+1] - y[k])
      f = f // 255
      if f == 2:
        if (image[i][j] == 0 and y[1] == 255 and y[2] == 255 and y[6] == 255 and y[7] == 255 and y[8] == 255 and y[3] == 0 and y[4] == 0 and y[5] == 0):
          print(i,j)
          temp[i,j] = [0,0,0]
  return temp

np.set_printoptions(threshold=np.inf)
base_dir = "result"
for i in range(3,4):
  path_test_image = os.path.join(base_dir, str(i)+".png")
  print(i)
  image_color = cv2.imread(path_test_image)
  image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
  adaptive_threshold = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
  new_image = find_border_point(adaptive_threshold,image_color)
  cv2.imshow('two-value', image_color)
  cv2.imshow('new',new_image)
  cv2.waitKey(0)