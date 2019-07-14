import os 
import cv2
import numpy as np
def image_binarization(image):#图片二值化
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    if(mean == 255):
      mean = 0
    ret, binary =  cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)#原来是150
    return binary
def is51W(list):
  return not ((list - [255,0,0,255,255,255,0,0,0,0]).any())
def is6W(list):
  return not ((list - [255,0,0,255,255,0,0,0,0,0]).any())
def is71W(list):
  return not ((list - [255,0,0,255,0,0,0,0,0,0]).any())
def is72W(list):
  return not ((list - [255,0,0,0,255,0,0,0,0,0]).any())
def is4W(list):
  return not ((list - [255,0,255,255,255,255,0,0,0,0]).any())
def is32W(list):
  return not ((list - [255,255,255,255,255,255,0,0,0,255]).any())
def is11B(list):
  return not ((list - [0,0,255,255,255,255,255,255,255,0]).any())
def is12B(list):
  return not ((list - [0,255,255,255,255,255,255,255,0,255]).any())
def is2B(list):
  return not ((list - [0,0,255,255,255,255,255,255,0,0]).any())
def is31B(list):
  return not ((list - [0,0,255,255,255,255,255,0,0,0]).any())
def is4B(list):
  return not ((list - [0,0,255,255,255,255,0,0,0,0]).any())
def is52B(list):
  return not ((list - [0,0,255,255,255,0,0,0,0,0]).any())
text = open('songti.txt','a')
# for root,dirs,files in os.walk("./chinese"):
#     print(len(files))
#     all_block = 0
#     for fn in files:
#         if (fn == '.DS_Store'):
#             continue
#         print(fn)
#         path = "./chinese/" + fn
#         image_origin = cv2.imread(path)
#         image = image_binarization(image_origin)
#         for j in range(1,image.shape[1]-1):
#             for i in range(1,image.shape[0]-1):
#                 if (not (image[i][j] - [0,0,0]).any()):
#                     all_block += 1
#     print(all_block / len(files))
average = 1471
for root,dirs,files in os.walk("./chinese"):
    print(len(files))
    for fn in files:
        print(fn)
        if (fn == '.DS_Store'):
            continue
        path = "./chinese/" + fn
        image_origin = cv2.imread(path)
        image = image_binarization(image_origin)
        # cv2.imshow(fn,image)
        # cv2.waitKey(0)
        toW = 0
        toB = 0
        block = 0
        for j in range(1,image.shape[1]-1):
            for i in range(1,image.shape[0]-1):
                if (not (image[i][j] - [0,0,0]).any()):
                    block += 1
        for i in range(1,image.shape[0]-1):
            for j in range(1,image.shape[1]-1):
                y = np.zeros(10,dtype=int)
                y[0] = image[i][j]
                y[1] = image[i-1][j-1]
                y[2] = image[i-1][j]
                y[3] = image[i-1][j+1]
                y[4] = image[i][j+1]
                y[5] = image[i+1][j+1]
                y[6] = image[i+1][j]
                y[7] = image[i+1][j-1]
                y[8] = image[i][j-1]
                y[9] = y[1]
                i1 = [0,3,4,5,6,7,8,1,2,3]
                y1 = np.zeros(10,dtype=int)
                for ii in range(0,10):
                    y1[ii] = y[i1[ii]]
                i2 = [0,5,6,7,8,1,2,3,4,5]
                y2 = np.zeros(10,dtype=int)
                for ii in i2:
                    y2[ii] = y[i2[ii]]
                i3 = [0,7,8,1,2,3,4,5,6,7]
                y3 = np.zeros(10,dtype=int)
                for ii in i3:
                    y3[ii] = y[i3[ii]]
                rotate = [[0,1,2,3,4,5,6,7,8,1],[0,3,4,5,6,7,8,1,2,3],[0,5,6,7,8,1,2,3,4,5],[0,7,8,1,2,3,4,5,6,7]]
                symmetry = [0,7,6,5,4,3,2,1,8,7]
                flag = 0
                for r in range(0,4):
                    temp_y = np.zeros(10,dtype=int)
                    for value in range(0,10):
                        temp_y[value] = y[rotate[r][value]]
                    if (is51W(temp_y) or is6W(temp_y) or is71W(temp_y) or is72W(temp_y) or is4W(temp_y) or is32W(temp_y)):
                        flag = 1
                    temp_array = np.zeros(10,dtype=int)
                    for value in range(0,10):
                        temp_array[value] = temp_y[value]
                    for value in range(0,10):
                        temp_y[value] = temp_array[symmetry[value]]
                    if (is51W(temp_y) or is6W(temp_y) or is71W(temp_y) or is72W(temp_y) or is4W(temp_y) or is32W(temp_y)):
                        flag = 1
                    if (flag == 1):
                        toW += 1
                flag = 0
                for r in range(0,4):
                    temp_y = np.zeros(10,dtype=int)
                    for value in range(0,10):
                        temp_y[value] = y[rotate[r][value]]
                    if (is11B(temp_y) or is12B(temp_y) or is2B(temp_y) or is31B(temp_y) or is4B(temp_y) or is52B(temp_y)):
                        flag = 1
                    temp_array = np.zeros(10,dtype=int)
                    for value in range(0,10):
                        temp_array[value] = temp_y[value]
                    for value in range(0,10):
                        temp_y[value] = temp_array[symmetry[value]]
                    if (is11B(temp_y) or is12B(temp_y) or is2B(temp_y) or is31B(temp_y) or is4B(temp_y) or is52B(temp_y)):
                        flag = 1
                    if (flag == 1):
                        toB += 1
        print((toB + toW) / average / 2)
        text.write(fn + " " + str(round((toB + toW) / average / 2,2)) + '\n')
text.close()
        