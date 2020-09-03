# coding: utf-8


import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

im = PIL.Image.new('RGB', (480, 270), '#333333')
# im.show()
draw = PIL.ImageDraw.Draw(im)


# 绘制线段

draw.line((0, 0) + im.size, fill='#FFFFFF')
draw.line((0, im.size[1], im.size[0], 0), fill='#FFFFFF')


# 绘制离散的点
draw.point([(2, 3), (3, 2), (1, 4)], fill='#FFFFFF')


# 绘制圆弧
draw.arc((100, 50, 379, 219), 0, 180, fill='#FFFFFF')
draw.chord((70, 50, 179, 119), 0, 180, fill='#FFFFFF')
draw.ellipse((150, 50, 349, 189), fill='#FF00FF')
draw.pieslice((100, 50, 379, 219), 0, 90, fill='#00FFFF')


# 绘制矩形
draw.rectangle((100, 15, 279, 69), fill='#FFFF00')


# 绘制多边形
draw.polygon([(200, 50), (80, 50), (240, 250)], fill='#0000FF')
draw.polygon([(300, 50), (280, 50), (240, 250),
    (270, 30), (240, 30)], fill='#FF0000')


# 绘制文字
font = PIL.ImageFont.truetype('consola', 14)
print(draw.textsize('Hello World!', font))
draw.text((192, 130), 'Hello World!', '#000000', font)

font = PIL.ImageFont.truetype('arial', 24)
print(draw.textsize('TEST PIL', font))
draw.text((242, 190), 'TEST PIL', '#FFFFFF', font)


im.show()
im.save('4pil_imagechops.bmp')

'''
(96, 11)
(105, 22)
[Finished in 2.7s]
'''


