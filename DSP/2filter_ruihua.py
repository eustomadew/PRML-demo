# coding: utf-8



import numpy as np
import PIL.Image
import scipy.misc
import scipy.signal


slide1 = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])
slide2 = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
])
slide3 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])
slide4 = np.array([
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1]
])
target = np.full((3, 3), 0)
target[1, 1] = 1


def convert_2d(r, window=None):
    # 滤波掩模
    if window is None:
        window = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
    s = scipy.signal.convolve2d(r, window, mode='same', boundary='symm')
    # 像素值如果大于 255 则取 255, 小于 0 则取 0
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i][j] = min(max(0, s[i][j]), 255)
    s = s.astype(np.uint8)
    return s

def convert_3d(r, w=None):
    s_dsplit = []
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        ss = convert_2d(rr, w)
        s_dsplit.append(ss)
    s = np.dstack(s_dsplit)
    return s


im = PIL.Image.open('./jp.jpg')
im_mat = np.asarray(im)
im_converted_mat = convert_3d(im_mat)

# im_converted_mat = convert_3d(im_mat, w=slide1)
# im_converted_mat = convert_3d(im_mat, w=slide2)
# im_converted_mat = convert_3d(im_mat, w=slide3)
# im_converted_mat = convert_3d(im_mat, w=slide4)

# im_converted_mat = convert_3d(im_mat, w=target-slide1)
# im_converted_mat = convert_3d(im_mat, w=target-slide2)
# im_converted_mat = convert_3d(im_mat, w=target-slide3)
# im_converted_mat = convert_3d(im_mat, w=target-slide4)

im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()

im_converted.save('./2_filter_ruihua.jpg', quality=95)
# im_converted.save('./2_filter_rh1.jpg', quality=95)
# im_converted.save('./2_filter_rh2.jpg', quality=95)
# im_converted.save('./2_filter_rh3.jpg', quality=95)
# im_converted.save('./2_filter_rh4.jpg', quality=95)

# im_converted.save('./2_filter_rh5.jpg', quality=95)
# im_converted.save('./2_filter_rh6.jpg', quality=95)
# im_converted.save('./2_filter_rh7.jpg', quality=95)
# im_converted.save('./2_filter_rh8.jpg', quality=95)

