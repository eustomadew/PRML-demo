# coding: utf-8


import math

import numpy as np
import PIL.Image
import PIL.ImageFilter
import scipy.misc
import scipy.signal

# 概率密度函数
def get_cv(r, sigma):
    return 1 / (2 * math.pi * sigma ** 2) * math.exp((-r**2) / (2 * sigma ** 2))

# 高斯滤波掩模
def get_window():
    # 模糊半径为 2, sigma 为 1.5
    radius, sigma = 2, 1.5
    window = np.zeros((radius * 2 + 1, radius * 2 + 1))
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            r = (i ** 2 + j ** 2) ** 0.5
            window[i + radius][j + radius] = get_cv(r, sigma)
    return window / np.sum(window)

def convert_2d(r):
    window = get_window()
    s = scipy.signal.convolve2d(r, window, mode='same', boundary='symm')
    return s.astype(np.uint8)

def convert_3d(r):
    s_dsplit = []
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        ss = convert_2d(rr)
        s_dsplit.append(ss)
    s = np.dstack(s_dsplit)
    return s

im = PIL.Image.open('./jp.jpg')
im_mat = np.asarray(im)
im_converted_mat = convert_3d(im_mat)
im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()

im_converted.save('./2_filter_gauss1.jpg', quality=95)



import PIL.Image
import PIL.ImageFilter

im = PIL.Image.open('./jp.jpg')
im = im.filter(PIL.ImageFilter.GaussianBlur(radius=2))
im.show()

im.save('./2_filter_gauss2.jpg', quality=95)



