# coding: utf-8


""" 高通滤波器

# 实验代码
import numpy as np
import PIL.Image
import scipy.misc


def convert_2d(r):
    r_ext = np.zeros((r.shape[0] * 2, r.shape[1] * 2))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r_ext[i][j] = r[i][j]

    r_ext_fu = np.fft.fft2(r_ext)
    r_ext_fu = np.fft.fftshift(r_ext_fu)

    # 截止频率为 20
    d0 = 20
    # 2 阶巴特沃斯
    n = 2
    # 频率域中心坐标
    center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
    h = np.empty(r_ext_fu.shape)
    # 绘制滤波器 H(u, v)
    for u in range(h.shape[0]):
        for v in range(h.shape[1]):
            duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
            if duv == 0:
                h[u][v] = 0
            else:
                h[u][v] = 1 / ((1 + (d0 / duv)) ** (2*n))

    s_ext_fu = r_ext_fu * h
    s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
    s_ext = np.abs(s_ext)
    s = s_ext[0:r.shape[0], 0:r.shape[1]]

    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i][j] = min(max(s[i][j], 0), 255)

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

"""








# COMPARISON

import numpy as np
import PIL.Image
import scipy.misc


def convert_2d_LPF(r, ctype='ILPF'):
    r_ext = np.zeros((r.shape[0] * 2, r.shape[1] * 2))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r_ext[i][j] = r[i][j]

    r_ext_fu = np.fft.fft2(r_ext)
    r_ext_fu = np.fft.fftshift(r_ext_fu)

    # 截止频率为 100
    d0 = 100  #ILPF,BLPF,GLPF
    # 2 阶巴特沃斯
    n = 2  #  #BLPF,
    #
    # 频率域中心坐标
    center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
    h = np.empty(r_ext_fu.shape)
    # 绘制滤波器 H(u,v)
    for u in range(h.shape[0]):
        for v in range(h.shape[1]):
            duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
            #
            if ctype == 'ILPF':
                h[u][v] = duv < d0
            elif ctype == 'BLPF':
                h[u][v] = 1 / ((1 + (duv / d0)) ** (2*n))
            elif ctype == 'GLPF':
                # h[u][v] = np.e ** (-duv**2 / d0 ** 2)
                h[u][v] = np.e ** (-duv**2 / d0 ** 2 /2)

    s_ext_fu = r_ext_fu * h
    s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
    s_ext = np.abs(s_ext)
    s = s_ext[0:r.shape[0], 0:r.shape[1]]

    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i][j] = min(max(s[i][j], 0), 255)

    return s.astype(np.uint8)


def convert_2d_HPF(r, ctype='BHPF'):
    r_ext = np.zeros((r.shape[0] * 2, r.shape[1] * 2))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            r_ext[i][j] = r[i][j]

    r_ext_fu = np.fft.fft2(r_ext)
    r_ext_fu = np.fft.fftshift(r_ext_fu)

    # 截止频率为 20
    d0 = 20  #BHPF,
    # 2 阶巴特沃斯
    n = 2  # #BHPF,
    #
    # 频率域中心坐标
    center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
    h = np.empty(r_ext_fu.shape)
    # 绘制滤波器 H(u,v)
    for u in range(h.shape[0]):
        for v in range(h.shape[1]):
            duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
            #
            if ctype == 'BHPF':
                if duv == 0:
                    h[u][v] = 0
                else:
                    h[u][v] = 1 / ((1 + (d0 / duv)) ** (2*n))
            elif ctype == 'IHPF':
                h[u][v] = duv > d0
            elif ctype == 'GHPF':
                h[u][v] = 1 - np.e ** (-duv**2 / d0 ** 2 /2)
                # h[u][v] = 1 - np.e ** (-duv**2 / d0 ** 2)

    s_ext_fu = r_ext_fu * h
    s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
    s_ext = np.abs(s_ext)
    s = s_ext[0:r.shape[0], 0:r.shape[1]]

    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i][j] = min(max(s[i][j], 0), 255)

    return s.astype(np.uint8)


def convert_3d(r, ctype='BHPF'):
    s_dsplit = []
    for d in range(r.shape[2]):
        rr = r[:, :, d]
        #
        if ctype.endswith('LPF'):
            ss = convert_2d_LPF(rr, ctype)
        elif ctype.endswith('HPF'):
            ss = convert_2d_HPF(rr, ctype)
        else:
            ss = rr
        #   #
        s_dsplit.append(ss)
    s = np.dstack(s_dsplit)
    return s


im = PIL.Image.open('./jp.jpg')
im_mat = np.asarray(im)
# im_converted_mat = convert_3d(im_mat)
# im_converted = PIL.Image.fromarray(im_converted_mat)
# im_converted.show()

for k in ['LPF', 'HPF']:
    for v in ['I', 'B', 'G']:
        ctype = v + k
        im_converted_mat = convert_3d(im_mat, ctype)
        im_converted = PIL.Image.fromarray(im_converted_mat)
        # im_converted.show()
        im_converted.save(
            './2_filter_pinyu_gt_{}.jpg'.format(ctype), quality=95)

# [Finished in 39.5s]  # B/2
# [Finished in 36.6s]  # B without /2
# incorrect, forget the hyper-parameter

# [Finished in 47.4s]


