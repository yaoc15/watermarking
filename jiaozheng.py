import os
import cv2
import math
import random
import numpy as np
from scipy import misc, ndimage

filepath = 'jiaozheng.png'

img = cv2.imread('jiaozheng.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imwrite('edges.png', edges)
#cv2.imshow('edges',edges)
#cv2.waitKey(0)
#霍夫变换
lines = cv2.HoughLines(edges,1,np.pi/180,0)
print(lines)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
t = float(y2-y1)/(x2-x1)
rotate_angle = math.degrees(math.atan(t))
if rotate_angle > 45:
  rotate_angle = -90 + rotate_angle
elif rotate_angle < -45:
  rotate_angle = 90 + rotate_angle
rotate_img = ndimage.rotate(img, rotate_angle)
#cv2.imshow('origin',img)
#cv2.imshow('haha',rotate_img)
#cv2.waitKey(0)

cv2.imwrite('after.png', rotate_img)