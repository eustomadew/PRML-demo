# coding: utf-8


# 图像的傅里叶变换与反变换
import numpy as np
import scipy.misc
import PIL.Image
import matplotlib.pyplot as plt

im = PIL.Image.open('./jp.jpg')
im = im.convert('L')
im_mat = np.asarray(im)
rows, cols = im_mat.shape

# 扩展 M * N 图像到 2M * 2N
im_mat_ext = np.zeros((rows * 2, cols * 2))
for i in range(rows):
    for j in range(cols):
        im_mat_ext[i][j] = im_mat[i][j]

# 傅里叶变换
im_mat_fu = np.fft.fft2(im_mat_ext)
# 将低频信号移植中间, 等效于在时域上对 f(x, y) 乘以 (-1)^(m + n)
im_mat_fu = np.fft.fftshift(im_mat_fu)

# 显示原图
plt.subplot(121)
plt.imshow(im_mat, 'gray')
plt.title('original')
plt.subplot(122)
# 在显示频率谱之前, 对频率谱取实部并进行对数变换
plt.imshow(np.log(np.abs(im_mat_fu)), 'gray')
plt.title('fourier')
plt.show()

# 傅里叶反变换
im_converted_mat = np.fft.ifft2(np.fft.ifftshift(im_mat_fu))
# 得到傅里叶反变换结果的实部
im_converted_mat = np.abs(im_converted_mat)
# 提取左上象限
im_converted_mat = im_converted_mat[0:rows, 0:cols]
# 显示图像
im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()


# ref:
# https://blog.csdn.net/z704630835/article/details/84968767
# https://blog.csdn.net/zhicheng_angle/article/details/88548518

print("mode:", im_converted.mode)

if im_converted.mode == 'F':
    im_converted = im_converted.convert("L")  # 转换成灰度图
elif im_converted.mode == 'P':
    im_converted = im_converted.convert("RGB")
im_converted.save('./2_filter_pinyu.jpeg', quality=95)


