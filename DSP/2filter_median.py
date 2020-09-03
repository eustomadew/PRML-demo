# coding: utf-8


import numpy as np
import PIL.Image
import scipy.misc
import scipy.ndimage

def convert_2d(r):
    n = 10
    s = scipy.ndimage.median_filter(r, (n, n))
    #
    # s = scipy.ndimage.minimum_filter(r, (n, n))
    # s = scipy.ndimage.maximum_filter(r, (n, n))
    # s = scipy.ndimage.percentile_filter(r, 50, (n, n))
    # s = scipy.ndimage.rank_filter(r, 3, (n, n))
    # s = scipy.ndimage.uniform_filter(r, (n, n))
    #
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

# im_converted.save('./2_filter_median_min.jpg', quality=95)
# im_converted.save('./2_filter_median_max.jpg', quality=95)
# im_converted.save('./2_filter_median_perc.jpg', quality=95)
# im_converted.save('./2_filter_median_rank.jpg', quality=95)
# im_converted.save('./2_filter_median_unif.jpg', quality=95)

im_converted.save('./2_filter_median_medi.jpg', quality=95)

