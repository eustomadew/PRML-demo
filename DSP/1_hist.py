# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import scipy.misc

def convert_2d(r):
    x = np.zeros([256])
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            x[r[i][j]] += 1
    x = x / r.size

    sum_x = np.zeros([256])
    for i, _ in enumerate(x):
        sum_x[i] = sum(x[: i])

    s = np.empty(r.shape, dtype=np.uint8)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            s[i][j] = 255 * sum_x[r[i][j]]
    return s

im = PIL.Image.open('./jp.jpg')
im = im.convert('L')
im_mat = np.asarray(im)

# 显示输入直方图
plt.hist(im_mat.reshape([im_mat.size]), 255, normed=1)
plt.show()

im_converted_mat = convert_2d(im_mat)

# 显示输出直方图
plt.hist(im_converted_mat.reshape([im_converted_mat.size]), 256, normed=1)
plt.show()

im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()
