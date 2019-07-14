from aip import AipOcr
import cv2
import sys
import os
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
IMAGE_PATH = './tmp/' + sys.argv[1]
np.set_printoptions(threshold=np.inf)
K = 0.15 #步长
def get_word_block_num():
  path='./baidu_result'
  count = 0 
  for fn in os.listdir(path):
    if (fn != '.DS_Store'):
      count = count + 1
  return count
WORD_BLOCK_NUM = get_word_block_num() #切分的字数个数

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
def find_border_point(image,image_color,t):
  temp = image_color.copy()
  tmp = t
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
      if (tmp > 0):
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
          temp[i][j] = [0,0,0]
          #temp[i][j] = [0,255,0]
          tmp -= 1
      if (tmp < 0):
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
          temp[i][j] = [255,255,255]
          #temp[i][j] = [0,0,255]
          tmp += 1
        # if (is11B(y) or is12B(y) or is2B(y) or is31B(y) or is4B(y) or is52B(y) or is11B(y1) or is12B(y1) or is2B(y1) or is31B(y1) or is4B(y1) or is52B(y1) or is11B(y2) or is12B(y2) or is2B(y2) or is31B(y2) or is4B(y2) or is52B(y2) or is11B(y3) or is12B(y3) or is2B(y3) or is31B(y3) or is4B(y3) or is52B(y3)):
        #   temp[i][j] = [255,255,255]
        #   tmp += 1
        #   a[i][j] = 255
  #print(tmp)
  return temp,tmp
def get_file_content(filePath): #读入图片文件
  with open(filePath,'rb') as fp:
    return fp.read()

def baidu_image_segmentation(): #调用百度ai的api划分文本图片上的文字
  APP_ID = '15546110'
  API_KEY = 'lNuZFA2Co3hHpwG6GM3iMSYA'
  SECRET_KEY = 'cGqgNQkEVNGugtDUjDM5tctP4lXIzDaN'
  client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
  image = get_file_content(IMAGE_PATH)
  options = {}
  options["recognize_granularity"] = "small"
  options["vertexes_location"] = "true"
  options["language_type"] = "JAP"
  #print(client.general(image,options))
  return client.general(image,options)

def get_position_list(): #处理baidu返回的结果生成包含每个字的方块数组
  segmentation_result = baidu_image_segmentation()
  #print(segmentation_result)
  #print(segmentation_result['words_result'])
  position_list = []
  for i in range(segmentation_result['words_result_num']):
    for j in range(len(segmentation_result['words_result'][i]['chars'])):
      position_list.append({
        'x':segmentation_result['words_result'][i]['chars'][j]['location']['top'],
        'y':segmentation_result['words_result'][i]['chars'][j]['location']['left'],
        'w':segmentation_result['words_result'][i]['chars'][j]['location']['height'],
        'h':segmentation_result['words_result'][i]['chars'][j]['location']['width']
      })
  return position_list

def get_words():
  segmentation_result = baidu_image_segmentation()
  #print(segmentation_result)
  word_list = []
  for i in range(segmentation_result['words_result_num']):
    for j in range(len(segmentation_result['words_result'][i]['chars'])):
      word_list.append(segmentation_result['words_result'][i]['chars'][j]['char'])
  print("word_list",word_list)
  return word_list

def get_base_words(t):
  filePath = "./character/" + t + ".txt"
  result = []
  for line in open(filePath):
    line = line.strip('\n')
    result.append(line)
  return result


def image_segmentation(): #读入图片进行文字划分
  image = cv2.imread(IMAGE_PATH)
  image_binary = image_binarization(image)
  position_list = get_position_list()
  print(position_list)
  for i in range(len(position_list)):
    x = position_list[i]['y']
    y = position_list[i]['x']
    w = position_list[i]['h']
    h = position_list[i]['w']
    x1 = x
    x2 = x + w 
    flag = 1
    tmp_image = image_binary[y+1:y+h,x1:x2]
    gray = cv2.cvtColor(image[y+1:y+h,x1:x2], cv2.COLOR_RGB2GRAY)
    cv2.imshow('gray',gray)
    cv2.imshow('tmp_image_first', tmp_image)
    x1 = x1 + 10
    x2 = x2 - 15
    while (flag == 1 and x1 >= x - 20):
      flag = 0
      x1 = x1 - 1
      for j in range(y,y+h):
        if (image_binary[j][x1] == 0):
          flag = 1
    flag = 1
    while (flag == 1 and x2 <= x + w + 30):
      flag = 0
      x2 = x2 + 1
      for j in range(y,y+h):
        if (image_binary[j][x2] == 0):
          flag = 1
    pt1 = (x1 - 1, y)
    pt2 = (x2, y + h)
    tmp_image = image_binary[y+1:y+h,x1:x2]
    #cv2.imshow('tmp_image', tmp_image)
    #cv2.waitKey(0)
    cv2.rectangle(image, pt1, pt2, (0, 0, 255))
  cv2.imshow('splited char image', image)
  cv2.imwrite("split.png",image)
  cv2.waitKey(0)

def save_image_block(): #读入图片,存储文字划分的结果
  image = cv2.imread(IMAGE_PATH)
  position_list = get_position_list()
  print(position_list)
  image_binary = image_binarization(image)
  for i in range(len(position_list)):
    x = position_list[i]['y']
    y = position_list[i]['x']
    w = position_list[i]['h']
    h = position_list[i]['w']
    x1 = x
    x2 = x + w 
    flag = 1
    x1 = x1 + 10
    x2 = x2 - 10
    while (flag == 1 and x1 >= x - 20):
      flag = 0
      x1 = x1 - 1
      for j in range(y,y+h):
        if (image_binary[j][x1] == 0):
          flag = 1
    flag = 1
    while (flag == 1 and x2 <= x + w + 30):
      flag = 0
      x2 = x2 + 1
      for j in range(y,y+h):
        if (image_binary[j][x2] == 0):
          flag = 1
    tmp_image = image[y+1:y+h,x1:x2]
    #tmp_image = image[y+1:y+h,x+1:x+w]
    cv2.imwrite("baidu_result/" + str(i) + ".png",tmp_image)

def image_binarization(image):#图片二值化
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary =  cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
    # h, w =gray.shape[:2]
    # m = np.reshape(gray, [1,w*h])
    # mean = m.sum()/(w*h)
    # if(mean == 255):
    #   mean = 0
    #ret, binary =  cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)#原来是150
    return binary

def count_black_pixel(image):#统计黑色像素点个数
  image_binary = image_binarization(image)
  num = 0
  for j in range(1,image.shape[1]-1):
    for i in range(1,image.shape[0]-1):
      if (not (image_binary[i][j] - [0,0,0]).any()):
        num += 1
  return num

def count_gray_pixel(image):#统计一个图像的灰度值和
  num = 0
  for j in range(1,image.shape[1]-1):
    for i in range(1,image.shape[0]-1):
      num += (255 - image[i][j])
  return num

def get_average_pixel():#二值图像统计平均像素点
  print('WORD_BLOCK_NUM',WORD_BLOCK_NUM)
  pixel = np.zeros(WORD_BLOCK_NUM,dtype=int)
  new_pixel = []
  average_pixel = 0
  new_average_pixel = 0
  word_list = get_words()
  for i in range(0,WORD_BLOCK_NUM):
    if (i == 128):
        break
    image_path = './baidu_result/' + str(i) + '.png'
    color_image = cv2.imread(image_path)
    #image_binary = image_binarization(color_image)
    pixel[i] = count_black_pixel(color_image)
    average_pixel += pixel[i]
  average_pixel = round(average_pixel / (WORD_BLOCK_NUM),2)
  base_word = []
  for line in open("./character/japan_jiance.txt"):
    line = line.strip('\n')
    base_word.append(line)
  print("base_word",base_word)
  for i in range(0,WORD_BLOCK_NUM):
    if (pixel[i] >= average_pixel*0.3):#原为0.3
    #if ((not str(i) in base_word)):
      #print(word_list[i])
      new_pixel.append(i)
      new_average_pixel += pixel[i]
  new_average_pixel = round(new_average_pixel / len(new_pixel),2)
  print('符合要求的字块数',len(new_pixel))
  print(new_pixel)
  return (new_pixel,new_average_pixel)

def get_average_gray_pixel():#灰度图像统计平均像素点
  pixel = np.zeros(WORD_BLOCK_NUM,dtype=int)
  new_pixel = []
  average_pixel = 0
  new_average_pixel = 0
  for i in range(0,WORD_BLOCK_NUM):
    image_path = './baidu_result/' + str(i) + '.png'
    color_image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    pixel[i] = count_gray_pixel(image_gray)
    average_pixel += pixel[i]
  average_pixel = round(average_pixel / (WORD_BLOCK_NUM),2)
  for i in range(0,WORD_BLOCK_NUM):
    if (pixel[i] >= average_pixel * 0.3):#原为0.3
      new_pixel.append(i)
      new_average_pixel += pixel[i]
  new_average_pixel = round(new_average_pixel / len(new_pixel),2)
  return (new_pixel,new_average_pixel)

def get_print_image_block(information):#将每个字二值化后嵌入水印信息
  number = 1#嵌入的水印信息0,1
  tot_change_pixel = 0
  change_pixel = 0
  word_num = 0
  pixel_list = []
  average_pixel = 0
  new_pixel_sum = 0
  pixel_sum = 0
  pixel_list, average_pixel = get_average_pixel()
  print('average_pixel',average_pixel)
  print('pixel_list',pixel_list)
  image = cv2.imread(IMAGE_PATH)
  position_list = get_position_list()
  image_binary = image_binarization(image)
  for i in range(len(position_list)):#去除过多的文本
    if (i == 128):
        break
    if (i in pixel_list):
      x = position_list[i]['y']
      y = position_list[i]['x']
      w = position_list[i]['h']
      h = position_list[i]['w']
      x1 = x
      x2 = x + w 
      flag = 1
      x1 = x1 + 10
      x2 = x2 - 15
      while (flag == 1 and x1 >= x -20):
          flag = 0
          x1 = x1 -1
          for j in range(y,y+h):
              if (image_binary[j][x1] == 0):
                  flag = 1
      flag = 1
      while (flag == 1 and x2 <= x + w + 30):
          flag = 0
          x2 = x2 + 1
          for j in range(y,y+h):
              if (image_binary[j][x2] == 0):
                  flag = 1
      tmp_image = image[y+1:y+h,x1:x2]
      image_binary_block = image_binarization(tmp_image)
      image_binary_pixel = count_black_pixel(tmp_image)
      flag = 0
    #   if(word_num < len(information)):
    #     t1 = 0.56 * average_pixel - image_binary_pixel
    #     t2 = 0.88 * average_pixel - image_binary_pixel
    #     t3 = 1.20 * average_pixel - image_binary_pixel
    #     t4 = 1.52 * average_pixel - image_binary_pixel
    #     min_change = 9999999
    #     if (abs(min_change) > abs(t1)):
    #       min_change = t1
    #     if (abs(min_change) > abs(t2)):
    #       min_change = t2
    #     if (abs(min_change) > abs(t3)):
    #       min_change = t3
    #     if (abs(min_change) > abs(t4)):
    #       min_change = t4
    #     change_pixel = int(min_change)
    #     word_num = word_num + 1
    #     tot_change_pixel -= change_pixel
    #     flag = 0
    #   else:
    #     change_pixel = round(tot_change_pixel / (len(pixel_list) - word_num),0)
    #     tot_change_pixel -= change_pixel
    #     word_num += 1
    #     flag = 1
    #   print(i,change_pixel,image_binary_pixel)

      #print(i,change_pixel,word_num,len(pixel_list))
      tmp = round(image_binary_pixel / average_pixel /K,1)
      xx1 = math.floor(tmp)
      xx2 = math.floor(tmp + 1)
      x = xx2
      flag = 0
      if (word_num < len(information)):
        if ((number == 1 and xx1 % 2 == 1) or(number == 0 and xx1 % 2 == 0)):
          x = xx1
        change_pixel = int(round(x * K * average_pixel - image_binary_pixel))
        word_num = word_num + 1
        tot_change_pixel -= change_pixel
        flag = 0
      else:
        change_pixel = round(tot_change_pixel / (len(pixel_list) - word_num),0)
        tot_change_pixel -= change_pixel
        word_num += 1
        flag = 1
      print(i,change_pixel,image_binary_pixel)
      #print('image_binary_pixel',image_binary_pixel)
      new_image,rest_pixel = find_border_point(image_binary_block,tmp_image,change_pixel)
      print("rest",rest_pixel)
      if (flag == 0):
        tot_change_pixel += rest_pixel
      else:
        tot_change_pixel += rest_pixel
      #new_image_binary_block = image_binarization(new_image)
      new_image_binary_pixel = count_black_pixel(new_image)
      pixel_sum += image_binary_pixel
      new_pixel_sum += new_image_binary_pixel
      #print(image_binary_pixel,new_image_binary_pixel )
      #print('pixel_sum',pixel_sum,new_pixel_sum)
      #cv2.imshow('origin',tmp_image)
      #cv2.imshow('new',new_image)
      for i1 in range(y+1,y+h):
        for i2 in range(x1,x2):
          image[i1][i2] = new_image[i1-y-1][i2-x1]
      #cv2.waitKey(0)
  cv2.imshow('image',image)
  #cv2.waitKey(0)
  print('tot_change_pixel',tot_change_pixel)
  print('pixel_sum',pixel_sum,new_pixel_sum)
  print('new_image_pixel',new_pixel_sum / len(pixel_list))
  cv2.imwrite("water_print/result.png",image)
  

def deposit_water_print_image_block():#提取每个字的水印信息
  pixel_list = []
  average_pixel = 0
  pixel_list, average_pixel = get_average_pixel()
  print('gray_pixel_list',average_pixel)
  print(pixel_list)
  suitable_num = 0
  odd_sum = 0
  for i in range(len(pixel_list)):
    image_path = './baidu_result/' + str(pixel_list[i]) + '.png'
    color_image = cv2.imread(image_path)
    # image_gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    # image_binary_pixel = count_gray_pixel(image_gray)
    #image_binary = image_binarization(color_image)
    image_binary_pixel = count_black_pixel(color_image)
    tmp = round(image_binary_pixel / average_pixel /K,0)    
    #tmp = round(image_binary_pixel / average_pixel / 0.08,0)
    print(pixel_list[i],tmp)
    if (tmp % 2 == 1 and i <= 90):
      odd_sum = odd_sum + 1
    #tmp = round((image_binary_pixel / average_pixel - 0.7),2)* 10
    #tmp = round((image_binary_pixel / average_pixel ),2)
    #print(image_binary_pixel)
    # if(tmp >= 7 and tmp <= 10):
    #   print(pixel_list[i],tmp)
    #   suitable_num += 1
    #print('image_binary_pixel',image_binary_pixel)
  #print('choose_num',suitable_num)
  print("odd_sum",odd_sum)

# def deposit_water_print_image_block_gray():#提取每个字的水印信息用灰度求值
#   pixel_list = []
#   average_pixel = 0
#   pixel_list, average_pixel = get_average_gray_pixel()
#   print('gray_pixel_list',average_pixel)
#   print(pixel_list)
#   for i in range(len(pixel_list)):
#     image_path = './baidu_result/' + str(pixel_list[i]) + '.png'
#     color_image = cv2.imread(image_path)
#     image_gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
#     image_binary_pixel = count_gray_pixel(image_gray)
#     #image_binary = image_binarization(color_image)
#     #image_binary_pixel = count_black_pixel(color_image)
#     tmp = round(image_binary_pixel / average_pixel /K,0)
#     print(pixel_list[i]+'\t'+tmp)
#     #print('image_binary_pixel',image_binary_pixel)
def image_denoise():
  img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
  img = cv2.multiply(img, 1.2)
  kernel = np.ones((1, 1), np.uint8)
  img = cv2.erode(img, kernel, iterations=1)
  cv2.imwrite('after.png',img)

def embed_watermark():
  #image_segmentation()
  save_image_block()
  data = []
  for i in range(0,89):
    data.append(1)
  get_print_image_block(data)#200bit信息

def deposit_watermark():
  #image_segmentation()
  save_image_block()
  deposit_water_print_image_block()

# word_list = get_words()
# base_word = get_base_words('0.1')
# for i in range(len(word_list)):
#   if word_list[i] in base_word:
#     print(word_list[i])
#image_denoise()
#image_segmentation()
deposit_watermark()
#embed_watermark()
# img = cv2.imread(IMAGE_PATH)
# img_binary = image_binarization(img)
# cv2.imshow('img_binary',img_binary)
# cv2.imwrite('after.png',img_binary)
# cv2.waitKey(0)
# median = cv2.medianBlur(img, 5)
# cv2.imwrite('median.png',median)
# image = cv2.imread(IMAGE_PATH)
# image_binary = image_binarization(image)
# cv2.imwrite("./result_binary.png",image_binary)