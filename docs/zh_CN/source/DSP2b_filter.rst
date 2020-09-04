
===================================
数值图像处理 - 频域滤波
===================================


-----------------------------------
频域滤波基础
-----------------------------------

傅里叶变换
===================================

傅里叶变换 (Fourier transform) 是一种线性的积分变换, 常在将信号在时域 (或空域) 和频域之间变换时使用, 在物理学和工程学中有许多应用. 因其基本思想首先由法国学者约瑟夫·傅里叶系统地提出, 所以以其名字来命名以示纪念.

经过傅里叶变换而生成的函数 :math:`\hat{f}` 称作原函数 :math:`f` 的傅里叶变换、亦或其频谱. 在许多情况下, 傅里叶变换是可逆的, 即可通过 :math:`\hat{f}` 得到其原函数 :math:`f` . 通常情况下, :math:`f` 是实数函数, 而 :math:`\hat{f}` 则是复数函数, 用一个复数来表示振幅和相位.

傅里叶变换将函数的时域 (红色) 与频域 (蓝色) 相关联. 频谱中的不同成分频率在频域中以峰值形式表示:

.. image:: http://accu.cc/img/pil/frequency_filter/fourier_transform_time_and_frequency_domains.gif
    :height: 240px
    :width: 300px

频域中的滤波基础
===================================

1. 将 M * N 大小的图像扩展到 2M * 2N, 多余像素以 0 填充
2. 用 :math:`(-1)^{M+N}` 乘以输入图像进行中心变换
3. 计算图像的 DFT, 即 :math:`F(u,v)`
4. 用滤波器函数 :math:`H(u,v)` 乘以 :math:`F(u,v)`
5. 计算 4 中结果的反 DFT
6. 得到 5 中结果的实部
7. 用 :math:`(-1)^{M+N}` 乘以 6 中的结果
8. 提取 7 中结果左上象限 :math:`M*N` 大小的区域

.. image:: http://accu.cc/img/pil/frequency_filter/step.jpg

:math:`F(u,v)` 的中心部分为低频信号, 边缘部分为高频信号. 高频信号保存了图像的细节.  :math:`H(u,v)` 也被称为滤波器. 输出图像的傅里叶变换为: 
:math:`G(u,v) = F(u,v) H(u,v)`

:math:`H` 与 :math:`F` 的相乘涉及二维函数, 并在逐元素的基础上定义. 即: :math:`H` 的第一个元素乘以 :math:`F` 的第一个元素, :math:`H` 的第二个元素乘以 :math:`F` 的第二个元素, 以此类推.

相关代码
===================================

::

    # 图像的傅里叶变换与反变换
    import numpy as np
    import scipy.misc
    import PIL.Image
    import matplotlib.pyplot as plt

    im = PIL.Image.open('/img/jp.jpg')
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

图像与其频率谱图像:

.. image:: http://accu.cc/img/pil/frequency_filter/image_and_its_frequency_spectrum.png

.. admonition:: 输出结果图片

    refs:

    - https://blog.csdn.net/z704630835/article/details/84968767
    - https://blog.csdn.net/zhicheng_angle/article/details/88548518

    ::

        print("mode:", im_converted.mode)

        if im_converted.mode == 'F':
            im_converted = im_converted.convert("L")  # 转换成灰度图
        elif im_converted.mode == 'P':
            im_converted = im_converted.convert("RGB")
        im_converted.save('./2_filter_pinyu.jpeg', quality=95)

    .. image:: ../../../DSP/2_filter_pinyu.jpeg

.. admonition:: OSError cannot write mode F as JPEG

    ref: https://www.jianshu.com/p/e8d058767dfa

    对于 PIL 模块来说，其模式包括以下几种：

    - 1 :     1 位像素，黑和白，存成 8 位的像素
    - L :     8 位像素，黑白
    - P :     8 位像素，使用调色板映射到任何其他模式
    - RGB :   3x8 位像素，真彩
    - RGBA :  4x8 位像素，真彩+透明通道
    - CMYK :  4x8 位像素，颜色隔离
    - YCbCr : 3x8 位像素，彩色视频格式
    - I :     32 位整型像素
    - F :     32 位浮点型像素

-----------------------------------
频域滤波 - 低通滤波
-----------------------------------

低通滤波
===================================

一幅图像的边缘和其他尖锐的灰度转换对其傅里叶变换的高频信号有贡献. 因此, 在频域平滑(模糊)可通过对高频信号的衰减来达到. 因为 :math:`F(u,v)` 的中心部分为低频信号, 边缘部分为高频信号, 如果将 :math:`F(u,v)` 边缘部分屏蔽, 那么就相当于进行了低通滤波.

考虑三种滤波器: 理想滤波器, 巴特沃斯滤波器和高斯滤波器.

理想低通滤波器
===================================

在以原点为圆心, :math:`D_0` 为半径的圆内, 无衰减的通过所有频率, 而在该圆外阻断所有频率的滤波器称为理想低通滤波器 (ILPF). 它由下面的函数所决定: 
:math:`H(u, v) =
\begin{cases}
1 \,, & D(u, v) < D_0 \\
0 \,, & D(u, v) >= D_0
\end{cases}`

其中, :math:`D_0` 为一个正常数 (称为截止频率), :math:`D(u,v)` 是频率域中心点 :math:`(u,v)` 与频率矩形中心的距离. 

.. image:: http://accu.cc/img/pil/frequency_filter_lpf/ilpf.jpg

::

    # 理想低通滤波器代码实现
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

        # 截止频率为 100
        d0 = 100
        # 频率域中心坐标
        center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
        h = np.empty(r_ext_fu.shape)
        # 绘制滤波器 H(u, v)
        for u in range(h.shape[0]):
            for v in range(h.shape[1]):
                duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
                h[u][v] = duv < d0

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

    im = PIL.Image.open('/DSP/jp.jpg')
    im_mat = np.asarray(im)
    im_converted_mat = convert_3d(im_mat)
    im_converted = PIL.Image.fromarray(im_converted_mat)
    im_converted.show()

.. image:: http://accu.cc/img/pil/frequency_filter_lpf/ilpf_sample.jpg

如上图所示, 使用理想低通滤波器可以看到明显的振铃状波纹, 因此应用中很少采用理想低通滤波器. 

巴特沃斯低通滤波器
===================================

截止频率位于距原点 :math:`D_0` 处的 n 阶巴特沃斯低通滤波器 (BLPF) 的传递函数为 
:math:`H(u, v) = \frac{1}{1 + [D(u, v) / D_0]^{2n}}`

.. image:: http://accu.cc/img/pil/frequency_filter_lpf/blpf.jpg

与 ILPF 不同, BLPF 传递函数并没有在通过频率与滤除频率之间给出明显截止的尖锐的不连续性. 对于具有平滑传递函数的滤波器, 可在这样一点上定义截止频率, 即使 :math:`H(u,v)` 下降为其最大值的某个百分比的点 (如 50%).

::

    # 将理想低通滤波器的 convert_2d 函数修改一下
    def convert_2d(r):
        r_ext = np.zeros((r.shape[0] * 2, r.shape[1] * 2))
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r_ext[i][j] = r[i][j]

        r_ext_fu = np.fft.fft2(r_ext)
        r_ext_fu = np.fft.fftshift(r_ext_fu)

        # 截止频率为 100
        d0 = 100
        # 2 阶巴特沃斯
        n = 2
        # 频率域中心坐标
        center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
        h = np.empty(r_ext_fu.shape)
        # 绘制滤波器 H(u, v)
        for u in range(h.shape[0]):
            for v in range(h.shape[1]):
                duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
                h[u][v] = 1 / ((1 + (duv / d0)) ** (2*n))

        s_ext_fu = r_ext_fu * h
        s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
        s_ext = np.abs(s_ext)
        s = s_ext[0:r.shape[0], 0:r.shape[1]]

        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                s[i][j] = min(max(s[i][j], 0), 255)

        return s.astype(np.uint8)

.. image:: http://accu.cc/img/pil/frequency_filter_lpf/blpf_sample.jpg

归功于这种滤波器在低频到高频之间的平滑过渡, BLPF 没有产生可见的振铃效果.

高斯低通滤波器
===================================

高斯低通滤波器 (GLPF) 的传递函数为 
:math:`H(u, v) = e^{-D^2(u, v) / 2D_0^2}`

其中, :math:`D_0` 是截止频率, 当 :math:`D(u,v) = D_0` 时候, GLPF 下降到最大值的 0.607 处.

.. image:: http://accu.cc/img/pil/frequency_filter_lpf/glpf.jpg

::

    # 将理想低通滤波器的 convert_2d 函数修改一下
    def convert_2d(r):
        r_ext = np.zeros((r.shape[0] * 2, r.shape[1] * 2))
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                r_ext[i][j] = r[i][j]

        r_ext_fu = np.fft.fft2(r_ext)
        r_ext_fu = np.fft.fftshift(r_ext_fu)

        # 截止频率为 100
        d0 = 100
        # 频率域中心坐标
        center = (r_ext_fu.shape[0] // 2, r_ext_fu.shape[1] // 2)
        h = np.empty(r_ext_fu.shape)
        # 绘制滤波器 H(u, v)
        for u in range(h.shape[0]):
            for v in range(h.shape[1]):
                duv = ((u - center[0]) ** 2 + (v - center[1]) ** 2) ** 0.5
                h[u][v] = np.e ** (-duv**2 / d0 ** 2)

        s_ext_fu = r_ext_fu * h
        s_ext = np.fft.ifft2(np.fft.ifftshift(s_ext_fu))
        s_ext = np.abs(s_ext)
        s = s_ext[0:r.shape[0], 0:r.shape[1]]

        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                s[i][j] = min(max(s[i][j], 0), 255)

        return s.astype(np.uint8)


这三种的对比总结
========================

.. admonition:: 图片效果

    :math:`F(u,v)` 的中心是低频信号，边缘部分为高频信号，若将其边缘部分屏蔽，就相当于进行了低通滤波。 
    其中 :math:`D_0` 为正常数 (即截止频率)，:math:`D(u,v)` 是频率域中心点 :math:`(u,v)` 与频率矩形中心的距离。

    .. image:: http://accu.cc/img/pil/frequency_filter_lpf/ilpf.jpg
    .. image:: http://accu.cc/img/pil/frequency_filter_lpf/blpf.jpg
    .. image:: http://accu.cc/img/pil/frequency_filter_lpf/glpf.jpg

    - 理想 低通滤波器 (ILPG): 
      :math:`H(u,v) = \begin{cases}
      1 \,, & D(u,v) < D_0 \,; \\
      0 \,, & D(u,v) \geqslant D_0
      \end{cases}`
    - 巴特沃斯 低通滤波器 (BLPF): 
      :math:`H(u,v) = \frac{1}{ 1+[ D(u,v) / D_0 ]^{2n} }`
    - 高斯 低通滤波器 (GLPF): 
      :math:`H(u,v) = e^{ -D^2(u,v) / 2D_0^2 }`

    .. image:: ../../../DSP/2_filter_pinyu_ditong1.jpg
    .. image:: ../../../DSP/2_filter_pinyu_ditong2.jpg
    .. image:: ../../../DSP/2_filter_pinyu_ditong3.jpg


-----------------------------------
频域滤波 - 高通滤波
-----------------------------------

高通滤波
===================================

在低通滤波中我们说明了通过衰减图像傅里叶变换的高频信号可以平滑图像. 因为边缘和其他灰度急剧变化的区域与高频分量有关, 所以图像的锐化可以通过在频率域的高通滤波实现.

一个高通滤波器是从给定的低通滤波器用下式得到: 
:math:`H_{HP}(u, v) = 1 - H_{LP}(u, v)`

其中 :math:`H_{LP}(u, v)` 是低通滤波器的传递函数. 同样的, 高通滤波器也有理想 (IHPF), 巴特沃斯 (BHPF) 和高斯高通滤波器 (GHPF). 三种高通滤波器传递函数如下表所示:

- 理想 
  :math:`H(u, v) = \begin{cases} 0 & D(u, v) \le D_0 \\ 1 & D(u, v) > D_0 \\ \end{cases}`
- 巴特沃斯 
  :math:`H(u, v) = \frac{1}{1 + [D_0 / D(u, v)]^{2n}}`
- 高斯 
  :math:`H(u, v) = 1 - e^{-D^2(u, v) / 2D_0^2}`

实验结果
===================================

使用 :math:`n=2` 阶, 截止频率为 20 的巴特沃斯高通滤波器处理后的结果如下:

.. image:: http://accu.cc/img/pil/frequency_filter_hpf/sample.jpg

::

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

    im = PIL.Image.open('/DSP/jp.jpg')
    im_mat = np.asarray(im)
    im_converted_mat = convert_3d(im_mat)
    im_converted = PIL.Image.fromarray(im_converted_mat)
    im_converted.show()


这三种的对比总结
===================

滤波器的传递函数

- 低通滤波器: :math:`H_{LP}(u,v)`
- 高通滤波器: :math:`H_{HP}(u,v) = 1- H_{LP}(u,v)`

1. 理想

  - ILPF: :math:`H(u,v) = \begin{cases}
    1 \,, &\text{ if } D(u,v) < D_0 \,;\\
    0 \,, &\text{ if } D(u,v) \ge D_0
    \end{cases}`
  - IHPF: :math:`H(u,v) = \begin{cases}
    0 \,, &\text{ if } D(u,v) \le D_0 \,;\\
    1 \,, &\text{ if } D(u,v) > D_0
    \end{cases}`

2. 巴特沃斯

  - BLPF: :math:`H(u,v) = \frac{1}{ 1+[ D_(u,v) / D_0 ]^{2n} }`
  - BHPF: :math:`H(u,v) = \frac{1}{ 1+[ D_0 / D(u,v) ]^{2n} }`

3. 高斯

  - GLPF: :math:`H(u,v) = e^{ -D^2(u,v) / 2D_0^2 }`
  - GHPF: :math:`H(u,v) = 1 - e^{ -D^2(u,v) / 2D_0^2 }`

巴特沃斯的公式推导 
:math:`\begin{aligned}
1-BLPF =& \frac{ [D/D_0]^{2n} }{ 1+[D/D_0]^{2n} } & \\
=& \frac{1}{ \left( 1+[D/D_0]^{2n} \right)\cdot [D_0/D]^{2n} } & \\
=& \frac{1}{ [D_0/D]^{2n} +1 } &= BHPF
\end{aligned}`

.. admonition:: 绘制结果对比

    - 低通滤波器: d0=100, n=2
    - 高通滤波器: d0=20, n=2

    .. image:: ../../../DSP/2_filter_pinyu_gt_ILPF.jpg
    .. image:: ../../../DSP/2_filter_pinyu_gt_BLPF.jpg
    .. image:: ../../../DSP/2_filter_pinyu_gt_GLPF.jpg

    .. image:: ../../../DSP/2_filter_pinyu_gt_IHPF.jpg
    .. image:: ../../../DSP/2_filter_pinyu_gt_BHPF.jpg
    .. image:: ../../../DSP/2_filter_pinyu_gt_GHPF.jpg



-----------------------------------
频域滤波 - 带阻和带通滤波
-----------------------------------

带阻和带通滤波
===================================

带阻滤波器 (BR) 传递函数:

- 理想: 
  :math:`H(u, v) = \begin{cases}
  0 \,,& \text{ if }\, D_0 - \frac{W}{2} \le D \le D_0 + \frac{W}{2} \\
  1 \,,& \text{ otherwise }
  \end{cases}`
- 巴特沃斯: 
  :math:`H(u, v) = \frac{1}{1 + \Big[\frac{DW}{D^2 - D_0^2}\Big]^{2n}}`
- 高斯: 
  :math:`H(u, v) = 1 - e^{-\Big[\frac{D^2 - D_0^2}{DW}\Big]^2}` 
- 高斯也见到有形式 :math:`H(u, v) = 1 - e^{-\frac{1}{2}\Big[\frac{D^2(u,v) - D_0^2}{D(u,v)W}\Big]^2}`

其中 :math:`W` 是带宽, :math:`D` 是 :math:`D(u,v)` 距离滤波中心的距离, :math:`D_0` 是截止频率, :math:`n` 是巴特沃斯滤波器的阶数.

一个带通滤波器 (BP) 的传递函数是: 
:math:`H_{BP}(u, v) = 1 - H_{BR}(u, v)`


**refs:**

- 频域滤波-带通/带阻滤波 https://www.cnblogs.com/laumians-notes/p/8600688.html
- 高通/带阻/陷波滤波器 https://www.cnblogs.com/fuhaots2009/p/3465149.html
- 选择性滤波 https://zhuanlan.zhihu.com/p/148623127
- 频域选择性滤波 (带通带阻滤波) https://zhuanlan.zhihu.com/p/149335127

- 巴特沃斯滤波器 维基百科 https://zh.wikipedia.org/wiki/%E5%B7%B4%E7%89%B9%E6%B2%83%E6%96%AF%E6%BB%A4%E6%B3%A2%E5%99%A8
- 巴特沃斯滤波器原理 http://www.360doc.com/content/19/0928/15/42387867_863717254.shtml
- 巴特沃斯 (Butterworth) 滤波器 https://blog.csdn.net/zhwzhaowei/article/details/71037196
- 高通,带阻与陷波滤波器 https://blog.csdn.net/thnh169/article/details/17201293
- 高斯高通滤波器 https://blog.csdn.net/vvickey11/article/details/51126039


