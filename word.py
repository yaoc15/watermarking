import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
def is51W(list):
  return not ((list - [0,255,255,0,0,0,255,255,255,255]).any())
def is6W(list):
  return not ((list - [0,255,255,0,0,255,255,255,255,255]).any())
def is71W(list):
  return not ((list - [0,255,255,0,255,255,255,255,255,255]).any())
def is72W(list):
  return not ((list - [0,255,255,255,0,255,255,255,255,255]).any())
def is4W(list):
  return not ((list - [0,255,0,0,0,0,255,255,255,255]).any())
def is32W(list):
  return not ((list - [0,0,0,0,0,0,255,255,255,0]).any())
def is11B(list):
  return not ((list - [255,255,0,0,0,0,0,0,0,255]).any())
def is12B(list):
  return not ((list - [255,0,0,0,0,0,0,0,255,0]).any())
def is2B(list):
  return not ((list - [255,255,0,0,0,0,0,0,255,255]).any())
def is31B(list):
  return not ((list - [255,255,0,0,0,0,0,255,255,255]).any())
def is4B(list):
  return not ((list - [255,255,0,0,0,0,255,255,255,255]).any())
def is52B(list):
  return not ((list - [255,255,0,0,0,255,255,255,255,255]).any())
def count_black_pixel(image):
  num = 0
  for j in range(1,image.shape[1]-1):
    for i in range(1,image.shape[0]-1):
      if (image[i][j] == 0):
        num += 1
  return num

def find_border_point(image,image_color):
  temp = image_color.copy()
  for j in range(1,image.shape[1]-1):
    for i in range(1,image.shape[0]-1):
      y = np.zeros(10,dtype=int)
      y[0] = image[i][j]
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
        if (is51W(y) or is6W(y) or is71W(y) or is72W(y) or is4W(y) or is32W(y)):
          print(i,j)
        # if (is11B(y) or is12B(y) or is2B(y) or is31B(y) or is4B(y) or is52B(y)):
        #   print(i,j)
        #   temp[i,j] = [0,0,0]
        # if (image[i][j] == 0 and y[1] == 255 and y[2] == 255 and y[6] == 255 and y[7] == 255 and y[8] == 255 and y[3] == 0 and y[4] == 0 and y[5] == 0):
        #   print(i,j)
        #   temp[i,j] = [0,0,0]
  return temp

np.set_printoptions(threshold=np.inf)
base_dir = "result"
pixel = np.zeros(117,dtype=int)
average_pixel = 0
for i in range(1,116):
  path_test_image = os.path.join(base_dir, str(i)+".png")
  image_color = cv2.imread(path_test_image)
  image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
  adaptive_threshold = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
  pixel[i] = count_black_pixel(adaptive_threshold)
  average_pixel += pixel[i]
  # print(i)
  # new_image = find_border_point(adaptive_threshold,image_color)
  # cv2.imshow('two-value', adaptive_threshold)
  # cv2.imshow('new',new_image)
  # cv2.waitKey(0)
average_pixel = round(average_pixel / 115,2)
watering_list = [1,0,1,1,1,0,1,0,0,0,1,0,0,1,0,1,1] #第一位不计入
for i in range(1,17):
  tmp = round(pixel[i] / average_pixel /0.15,1)
  x1 = math.floor(tmp)
  x2 = math.floor(tmp + 1)
  x = x2
  if ((watering_list[i]  == 1 and x1 % 2 == 1) or(watering_list[i]  == 0 and x1 % 2 == 0)):
    x = x1
  change_pixel = int(round(x * 0.15 * average_pixel - pixel[i]))
  print(watering_list[i],tmp,x,change_pixel)