# coding: utf-8


""" 补色

import numpy as np
import PIL.Image
import scipy.misc

im = PIL.Image.open('./jp.jpg')
im = im.convert('RGB')
im_mat = np.asarray(im)

im_converted_mat = np.zeros_like(im_mat, dtype=np.uint8)
for x in range(im_mat.shape[0]):
    for y in range(im_mat.shape[1]):
        # 补色的公式是 max(r, g, b) + min(r, g, b) - [r, g, b]
        maxrgb = im_mat[x][y].max()
        minrgb = im_mat[x][y].min()
        im_converted_mat[x][y] = (int(maxrgb) + int(minrgb)) * np.ones(3) - im_mat[x][y]

im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()

"""


""" 反色

import numpy as np
import PIL.Image
import scipy.misc

im = PIL.Image.open('./jp.jpg')
im = im.convert('RGB')
im_mat = np.asarray(im)
# 反色的公式是 [255, 255, 255] - [r, g, b]
im_converted_mat = np.ones_like(im_mat) * 255 - im_mat
im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()

"""




""" 水印

import PIL.Image
import scipy.misc
import imageio

# im = scipy.misc.imread('./jp.jpg', mode='RGBA')
# im_water = scipy.misc.imread('./watermark.jpg', mode='RGBA')

im = imageio.imread('./jp.jpg', pilmode='RGBA')
im_water = imageio.imread('./watermark.png', pilmode='RGBA')

for x in range(im_water.shape[0]):
    for y in range(im_water.shape[1]):
        a = 0.3 * im_water[x][y][-1] / 255
        im[x][y][0:3] = (1 - a) * im[x][y][0:3] + a * im_water[x][y][0:3]

# PIL.Image.fromarray(im).show()

im_mat = PIL.Image.fromarray(im)
im_mat.show()
im_mat = im_mat.convert('RGB')
im_mat.save('./3other_watermark.jpg', quality=95)

# [Finished in 23.1s]

"""



""" LSB 不可见水印

import PIL.Image
import numpy as np
import scipy.misc
import imageio

# im = scipy.misc.imread('./jp.jpg', mode='RGBA')
# im_water = scipy.misc.imread('./water.jpg', mode='RGBA')
im = imageio.imread('./jp.jpg', pilmode='RGBA')
im_water = imageio.imread('./watermark.png', pilmode='RGBA')

# LSB 水印的第一步是滤除衬底最后 2 个低阶比特位
im = im // 4 * 4

for x in range(im_water.shape[0]):
    for y in range(im_water.shape[1]):
        im[x][y] += im_water[x][y] // 64

# 显示加水印后的图像
PIL.Image.fromarray(im.astype(np.uint8)).show()

im = im % 4 / 3 * 255
# 显示提取的水印图像
# PIL.Image.fromarray(im.astype(np.uint8)).show()


im_mat = PIL.Image.fromarray(im.astype(np.uint8))
im_mat.show()
im_mat = im_mat.convert('RGB')
im_mat.save('./3other_lsbwm.jpg', quality=95)

"""



""" 最近邻插值法

import PIL.Image

im = PIL.Image.open('./jp.jpg')
im_resized = PIL.Image.new(im.mode, (480, 270))
for r in range(im_resized.size[1]):
    for c in range(im_resized.size[0]):
        rr = round((r+1) / im_resized.size[1] * im.size[1]) - 1
        cc = round((c+1) / im_resized.size[0] * im.size[0]) - 1
        im_resized.putpixel((c, r), im.getpixel((cc, rr)))
im_resized.show()

"""


import PIL.Image

im = PIL.Image.open('./jp_ghost.bmp')
im = im.resize((im.size[0] // 2, im.size[1] // 2), PIL.Image.NEAREST)
im.show()





