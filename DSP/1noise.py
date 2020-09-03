# coding: utf-8



"""


import numpy as np
import matplotlib.pyplot as plt

# 生成均值为 0，标准差为 64 的正态分布数据
data = np.random.normal(0, 64, 1024 * 8)

# 在 plt 中画出直方图
plt.hist(data, 256, normed=1)
plt.show()



import PIL.Image
import scipy.misc
import numpy as np

def convert_2d(r):
    # 添加均值为 0，标准差为 64 的加性高斯白噪声
    s = r + np.random.normal(0, 64, r.shape)
    if np.min(s) >= 0 and np.max(s) <= 255:
        return s
    # 对比拉伸
    s = s - np.full(s.shape, np.min(s))
    s = s * 255 / np.max(s)
    s = s.astype(np.uint8)
    return s

def convert_3d(r):
    s_dsplit = []
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        ss = convert_2d(rr)
        s_dsplit.append(ss)
    s = np.dstack(s_dsplit)
    return s

im = PIL.Image.open('./jp.jpg')
im = im.convert('RGB')
im_mat = np.asarray(im)
im_converted_mat = convert_3d(im_mat)
im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()



"""




import PIL.Image
import scipy.misc
import numpy as np

def convert_2d(r):
    # 添加均值为 0，标准差为 64 的加性高斯白噪声
    s = r + np.random.normal(0, 64, r.shape)
    if np.min(s) >= 0 and np.max(s) <= 255:
        return s
    # 对比拉伸
    s = s - np.full(s.shape, np.min(s))
    s = s * 255 / np.max(s)
    s = s.astype(np.uint8)
    return s

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

k = 128
k = 16
k = 4
k = 64
k = 32

im_converted_mat = np.zeros(im_mat.shape)
for i in range(k):
    im_converted_mat += convert_3d(im_mat)

im_converted_mat = im_converted_mat / k
im_converted_mat = im_converted_mat - np.full(im_converted_mat.shape, np.min(im_converted_mat))
im_converted_mat = im_converted_mat * 255 / np.max(im_converted_mat)
im_converted_mat = im_converted_mat.astype(np.uint8)

im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()
im_converted.save('1_noise_k{}.jpg'.format(k), quality=75)




