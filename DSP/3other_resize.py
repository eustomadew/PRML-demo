# coding: utf-8


"""

import numpy as np
import PIL.Image
import scipy.misc

im = PIL.Image.open('./jp.jpg')
im_mat = np.asarray(im)
im_mat_resized = np.empty((270, 480, im_mat.shape[2]), dtype=np.uint8)

for r in range(im_mat_resized.shape[0]):
    for c in range(im_mat_resized.shape[1]):
        rr = (r + 1) / im_mat_resized.shape[0] * im_mat.shape[0] - 1
        cc = (c + 1) / im_mat_resized.shape[1] * im_mat.shape[1] - 1

        rr_int = int(rr)
        cc_int = int(cc)

        if rr == rr_int and cc == cc_int:
            p = im_mat[rr_int][cc_int]
        elif rr == rr_int:
            p = im_mat[rr_int][cc_int] * (cc_int + 1 - cc) + im_mat[rr_int][cc_int + 1] * (cc - cc_int)
        elif cc == cc_int:
            p = im_mat[rr_int][cc_int] * (rr_int + 1 - rr) + im_mat[rr_int + 1][cc_int] * (rr - rr_int)
        else:
            p11 = (rr_int, cc_int)
            p12 = (rr_int, cc_int + 1)
            p21 = (rr_int + 1, cc_int)
            p22 = (rr_int + 1, cc_int + 1)

            dr1 = rr - rr_int
            dr2 = rr_int + 1 - rr
            dc1 = cc - cc_int
            dc2 = cc_int + 1 - cc

            w11 = dr2 * dc2
            w12 = dr2 * dc1
            w21 = dr1 * dc2
            w22 = dr1 * dc1

            p = im_mat[p11[0]][p11[1]] * w11 + im_mat[p21[0]][p21[1]] * w12 + \
                im_mat[p12[0]][p12[1]] * w21 + im_mat[p22[0]][p22[1]] * w22

        im_mat_resized[r][c] = p


im_resized = PIL.Image.fromarray(im_mat_resized)
im_resized.show()


"""





'''

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

zen = """The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!"""

font = PIL.ImageFont.truetype('consola', 14)

im = PIL.Image.new('RGB', (552, 294), '#FFFFFF')
dr = PIL.ImageDraw.Draw(im)
dr.text((0, 0), zen, '#000000', font)

im.show()

'''


import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageStat

font = PIL.ImageFont.truetype('consola', 14)

im = PIL.Image.open('./jp.jpg')
im = im.convert('F')
size = im.size

rx = im.size[0]
ry = int(rx / size[0] * size[1] * 8 / 14)
im = im.resize((rx, ry), PIL.Image.NEAREST)

mean = PIL.ImageStat.Stat(im).mean[0]

words = []
for y in range(im.size[1]):
    for x in range(im.size[0]):
        p = im.getpixel((x, y))
        if p < mean / 2:
            c = '#'
        elif mean / 2 <= p < mean:
            c = '='
        elif mean <= p < mean + (255 - mean) / 2:
            c = '-'
        elif mean + (255 - mean) / 2 <= p:
            c = ' '
        else:
            raise ValueError(p)
        words.append(c)
    words.append('\n')

im.close()

im = PIL.Image.new('RGB', (im.size[0] * 8, im.size[1] * 14), '#FFFFFF')
dr = PIL.ImageDraw.Draw(im)
dr.text((0, 0), ''.join(words), '#000000', font)
im = im.resize(size, PIL.Image.LANCZOS)
# im.show()

im.save('./3other_asciiplot.jpg', quality=95)

