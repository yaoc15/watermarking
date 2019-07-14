import os
import pygame
from PIL import Image
chinese_dir = 'english'
if not os.path.exists(chinese_dir):
    os.mkdir(chinese_dir)

pygame.init()
start,end = (0x61, 0x7b) # 汉字编码范围
for codepoint in range(int(start), int(end)):
    word = chr(codepoint)
    print(word)
    font = pygame.font.Font("Arial.ttf", 64)
    rtext = font.render(word, True, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, os.path.join(chinese_dir, word + ".png"))
    img = Image.open("./english/" + word + ".png")
    clrs = img.getcolors()
    #print(len(clrs))
    if len(clrs) == 1:
      os.remove("./english/" + word + ".png")
