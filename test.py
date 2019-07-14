#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges

base_dir = "tmp"
path_test_image = os.path.join(base_dir, "test.png")
image_color = cv2.imread(path_test_image)
#new_shape = (image_color.shape[1] * 2, image_color.shape[0] * 2)
#image_color = cv2.resize(image_color, new_shape)
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)


adaptive_threshold = cv2.adaptiveThreshold(
    image,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY_INV, 11, 2)

#cv2.imshow('binary image', adaptive_threshold)
#cv2.waitKey(0)

horizontal_sum = np.sum(adaptive_threshold, axis=1)

plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
plt.gca().invert_yaxis()
#plt.show()

peek_ranges = extract_peek_ranges_from_array(horizontal_sum)
line_seg_adaptive_threshold = np.copy(adaptive_threshold)
for i, peek_range in enumerate(peek_ranges):
    x = 0
    y = peek_range[0]
    w = line_seg_adaptive_threshold.shape[1]
    h = peek_range[1] - y
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)
#cv2.imshow('line image', line_seg_adaptive_threshold)
#cv2.waitKey(0)


vertical_peek_ranges2d = []
for peek_range in peek_ranges:
    start_y = peek_range[0]
    end_y = peek_range[1]
    line_img = adaptive_threshold[start_y:end_y, :]
    vertical_sum = np.sum(line_img, axis=0)
    vertical_peek_ranges = extract_peek_ranges_from_array(
        vertical_sum,
        minimun_val=40,
        minimun_range=1)
    vertical_peek_ranges2d.append(vertical_peek_ranges)
0
## Draw
# color = (0, 255, 0)
# for i, peek_range in enumerate(peek_ranges):
#     for vertical_range in vertical_peek_ranges2d[i]:
#         x = vertical_range[0]
#         y = peek_range[0]
#         w = vertical_range[1] - x
#         h = peek_range[1] - y
#         pt1 = (x, y)
#         pt2 = (x + w, y + h)
#         cv2.rectangle(image_color, pt1, pt2, color)
#cv2.imshow('char image', image_color)
#cv2.waitKey(0)


def median_split_ranges(peek_ranges):
    new_peek_ranges = []
    widthes = []
    print(peek_ranges)
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    print('median',median_w)
    for i, peek_range in enumerate(peek_ranges):
        num_char = int(round(widthes[i]/median_w, 0))
        if num_char > 1:
            char_w = float(widthes[i] / num_char)
            for i in range(num_char):
                start_point = peek_range[0] + int(i * char_w)
                end_point = peek_range[0] + int((i + 1) * char_w)
                new_peek_ranges.append((start_point, end_point))
        elif i >0 and int(round((widthes[i-1]+widthes[i])/median_w)) == 1 and peek_ranges[i][0] - peek_ranges[i-1][1]<=4:
            print(i)
            new_peek_ranges.pop()
            start_point = peek_ranges[i-1][0]
            end_point = peek_ranges[i][1]
            print((start_point,end_point))
            widthes[i] = end_point - start_point + 1
            new_peek_ranges.append((start_point,end_point))
        else:
            new_peek_ranges.append(peek_range)
    print(new_peek_ranges)
    return new_peek_ranges


vertical_peek_ranges2d = []
for peek_range in peek_ranges:
    start_y = peek_range[0]
    end_y = peek_range[1]
    line_img = adaptive_threshold[start_y:end_y, :]
    cv2.imshow('line image', line_img)
    cv2.waitKey(0)
    vertical_sum = np.sum(line_img, axis=0)
    vertical_peek_ranges = extract_peek_ranges_from_array(vertical_sum,minimun_val=40,minimun_range=1)
    vertical_peek_ranges = median_split_ranges(vertical_peek_ranges)
    vertical_peek_ranges2d.append(vertical_peek_ranges)

## Draw
print('final',vertical_peek_ranges2d)
color = (0, 0, 255)
median = 0
width_sum = 0
width_num = 0
for i, peek_range in enumerate(peek_ranges):
    for vertical_range in vertical_peek_ranges2d[i]:
        width_sum += vertical_range[1] - vertical_range[0]
        width_num += 1
median = width_sum / width_num
print(median)
num_value = 1
for i, peek_range in enumerate(peek_ranges):
    for vertical_range in vertical_peek_ranges2d[i]:
        x = vertical_range[0]
        y = peek_range[0]
        w = vertical_range[1] - x
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        # if w > 0.5 * median:
        cv2.rectangle(image_color, pt1, pt2, color)
        tmp_image = image_color[y+1:y+h,x+1:x+w]
        cv2.imwrite("result/" + str(num_value) + ".png",tmp_image)
        num_value += 1
cv2.imshow('splited char image', image_color)
cv2.waitKey(0)