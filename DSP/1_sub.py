# coding: utf-8

# 取得原图与其高斯模糊图像的差值图像
import PIL.Image
import PIL.ImageFilter
import scipy.misc
import imageio
import numpy as np

def convert_2d(r, h):
    # 矩阵减法
    s = r - h
    if np.min(s) >= 0 and np.max(s) <= 255:
        return s
    # 线性拉伸
    s = s - np.full(s.shape, np.min(s))
    s = s * 255 / np.max(s)
    s = s.astype(np.uint8)
    return s

def convert_3d(r, h):
    s_dsplit = []
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        hh = h[:, :, d]
        ss = convert_2d(rr, hh)
        s_dsplit.append(ss)
    s = np.dstack(s_dsplit)
    return s

im = PIL.Image.open('./jp.jpg')
im = im.convert('RGB')
im_mat = np.asarray(im)
# 高斯模糊
im_converted = im.filter(PIL.ImageFilter.GaussianBlur(radius=2))
# im_converted_mat = scipy.misc.fromimage(im_converted)
im_converted_mat = np.asarray(im_converted)
im_converted.save('./1_sub.jpg', quality=95)

im_sub_mat = convert_3d(im_mat, im_converted_mat)
im_sub = PIL.Image.fromarray(im_sub_mat)
im_sub.show()

