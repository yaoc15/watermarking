# import codecs
# start,end = (0x4E00, 0x9FA5)  #汉字编码的范围
# with codecs.open("chinese.txt", "wb", encoding="utf-8") as f:
#  for codepoint in range(int(start),int(end)):
#   f.write(chr(codepoint))  #写出汉字

#encoding: utf-8
import os
import pygame
from PIL import Image
# img = Image.open("2.png")
# clrs = img.getcolors()
# print(len(clrs))
chinese_dir = 'chinese'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

pygame.init()
start,end = (0x4E00, 0x9FA5) # 汉字编码范围
for codepoint in range(int(start), int(end)):
    word = chr(codepoint)
    font = pygame.font.Font("Songti.ttc", 64)
    rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))
    img = Image.open("./chinese/" + word + ".png")
    clrs = img.getcolors()
    #print(len(clrs))
    if len(clrs) == 1:
      os.remove("./chinese/" + word + ".png")
