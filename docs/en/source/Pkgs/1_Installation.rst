==============
Installation
==============


------------------------
CUDA+cuDNN
------------------------

NVIDIA cuDNN  

- Download: https://developer.nvidia.com/rdp/cudnn-download 
- Archive: https://developer.nvidia.com/rdp/cudnn-archive 

=========== ==== ==== ==== ==== === === === === === === === ================
*NVIDIA*                    CUDA                             Notice: 
----------- ----------------------------------------------- ----------------
 cuDNN      11.0 10.2 10.1 10.0 9.2 9.1 9.0 8.0 7.5 7.0 6.5  Date
=========== ==== ==== ==== ==== === === === === === === === ================
 v8.0.2       X    X    X                                      Jul 24, 2020 
 v8.0.1 RC2   X    X                                           Jun 26, 2020 
 v7.6.5            X    X    X   X       X                   Nov 18/5, 2019 
 v7.6.4                 X    X   X       X                     Sep 27, 2019 
 v7.6.3                 X    X   X       X                     Aug 23, 2019 
 v7.6.2                 X    X   X       X                     Jul 22, 2019 
 v7.6.1                 X    X   X       X                     Jun 24, 2019 
 v7.6.0                 X    X   X       X                     May 20, 2019 
 v7.5.1                 X    X   X       X                     Apr 22, 2019 
 v7.5.0                 X    X   X       X                  Feb 25/21, 2019 
 v7.4.2                      X   X       X                     Dec 14, 2018 
 v7.4.1                      X   X       X                      Nov 8, 2018 
 v7.3.1                      X   X       X                     Sep 28, 2018 
 v7.3.0                      X           X                     Sep 19, 2018 
 v7.2.1                          X                              Aug 7, 2018 
 v7.1.4                          X       X   X                 May 16, 2018 
 v7.1.3                              X   X   X                 Apr 17, 2018 
 v7.1.2                          X   X   X                     Mar 21, 2018 
 v7.0.5                              X   X   X               Dec 11/5, 2017 
 v7.0.4                                  X                     Nov 13, 2017 
 v6.0                                        X   X             Apr 27, 2017 
 v5.1                                        X   X             Jan 20, 2017 
 v5                                          X   X             May 27, 2016 
 v4                                                  X         Feb 10, 2016 
 v3                                                  X          Sep 8, 2015 
 v2                                                      X     Mar 17, 2015 
 v1                                                      X   (cuDNN 6.5 R1) 
=========== ==== ==== ==== ==== === === === === === === === ================


------------------
TensorRT
------------------

NVIDIA TensorRT 

- Intro: https://developer.nvidia.com/zh-cn/tensorrt
- Download: https://developer.nvidia.com/nvidia-tensorrt-download


TensorRT for Linux x86 / Windows

- **+C:** CentOS/RedHat 7
- **+W:** Windows 10

======== ======== ===== ===== ===== == == ==== ==== ==== ==== === === === === ===
*NVIDIA*          Operation System         *NVIDIA*
----------------- ----------------------- ---------------------------------------
TensorRT          Ubuntu            +C +W  CUDA
----------------- ----------------- -- -- ---------------------------------------
License+ Version  18.04 16.04 14.04 7  10 11.0 10.2 10.1 10.0 9.2 9.1 9.0 8.0 7.5
======== ======== ===== ===== ===== == == ==== ==== ==== ==== === === === === ===
7.1 GA   7.1.3.4   X+    X+         X+      X    X 
7.1 GA   \                             X+   X 
7.0      7.0.0.11  X+                            X         X 
7.0      7.0.0.11        X+         X+           X         X           X 
7.0      \                             X+        X         X           X 
6.0 GA   6.0.1.8   X+    X+         X+           X 
6.0 GA   6.0.1.5   X+                                 X    X 
6.0 GA   6.0.1.5         X+    X+   X+                X    X           X 
6.0 GA   \                             X+             X    X           X 
5.1 GA   5.1.5.0   X+                                 X    X 
5.1 GA   5.1.5.0         X+    X+   X+                X    X           X 
5.1 GA   \                             X+             X    X           X 
5.1 RC   5.1.2.2   X+                                 X    X 
5.1 RC   5.1.2.2         X+    X+   X+                X    X           X 
5.1 RC   \                             X+             X    X           X 
5.0 GA   5.0.2.6   X+                                      X 
5.0 GA   5.0.2.6         X+    X+   X+                     X           X 
5.0 GA   \                             X+                  X           X 
4.0      4.0.1.6         X+    X+                              X       X   X 
3.0      3.0.4           X+    X+                                  X   X   X 
2.1      \               X+                                                X 
2.1      \                     X+                                          X   X 
1.0      Jan 2017              X+                                          X   X 
======== ======== ===== ===== ===== == == ==== ==== ==== ==== === === === === ===


-------------------------
TensorFlow
-------------------------

**Dependencies**

- Official Dependency Refers: https://www.tensorflow.org/install/source
- GPU Version Requirements: https://zhuanlan.zhihu.com/p/60924644
- Official GPU Support: https://www.tensorflow.org/install/gpu?hl=zh-cn
- CSDN· cuda with tfgpu: https://blog.csdn.net/qq_31347869/article/details/89060087
- CSDN· Corresponding tf,cuda,cudnn: https://blog.csdn.net/yuejisuo1948/article/details/81043962

**Verified Configurations**

================== =============== ============== =========== ============== ====== ======
  *Linux*                           CPU \& GPU                                   GPU 
---------------------------------- ----------------------------------------- -------------
 CPU Version         GPU Version   Python Version   Compiler    Build Tools   cuDNN  CUDA
================== =============== ============== =========== ============== ====== ======
tensorflow-2.1.0   tf-2.1.0         2.7, 3.5-3.7   GCC 7.3.1   Bazel 0.27.1   7.6    10.1 
tensorflow-2.0.0   tf-2.0.0         2.7, 3.3-3.7   GCC 7.3.1   Bazel 0.26.1   7.4    10.0 
tensorflow-1.14.0  tf\_gpu-1.14.0   2.7, 3.3-3.7   GCC 4.8     Bazel 0.24.1   7.4    10.0 
tensorflow-1.13.1  tf\_gpu-1.13.1   2.7, 3.3-3.7   GCC 4.8     Bazel 0.19.2   7.4    10.0 
tensorflow-1.12.0  tf\_gpu-1.12.0   2.7, 3.3-3.6   GCC 4.8     Bazel 0.15.0   7      9 
tensorflow-1.11.0  tf\_gpu-1.11.0   2.7, 3.3-3.6   GCC 4.8     Bazel 0.15.0   7      9 
tensorflow-1.10.0  tf\_gpu-1.10.0   2.7, 3.3-3.6   GCC 4.8     Bazel 0.15.0   7      9 
tensorflow-1.9.0   tf\_gpu-1.9.0    2.7, 3.3-3.6   GCC 4.8     Bazel 0.11.0   7      9 
tensorflow-1.8.0   tf\_gpu-1.8.0    2.7, 3.3-3.6   GCC 4.8     Bazel 0.10.0   7      9 
tensorflow-1.7.0   \                2.7, 3.3-3.6   GCC 4.8     Bazel 0.10.0   
\                  tf\_gpu-1.7.0    2.7, 3.3-3.6   GCC 4.8     Bazel 0.9.0    7      9 
tensorflow-1.6.0   tf\_gpu-1.6.0    2.7, 3.3-3.6   GCC 4.8     Bazel 0.9.0    7      9 
tensorflow-1.5.0   tf\_gpu-1.5.0    2.7, 3.3-3.6   GCC 4.8     Bazel 0.8.0    7      9 
tensorflow-1.4.0   tf\_gpu-1.4.0    2.7, 3.3-3.6   GCC 4.8     Bazel 0.5.4    6      8 
tensorflow-1.3.0   tf\_gpu-1.3.0    2.7, 3.3-3.6   GCC 4.8     Bazel 0.4.5    6      8 
tensorflow-1.2.0   tf\_gpu-1.2.0    2.7, 3.3-3.6   GCC 4.8     Bazel 0.4.5    5.1    8 
tensorflow-1.1.0   tf\_gpu-1.1.0    2.7, 3.3-3.6   GCC 4.8     Bazel 0.4.2    5.1    8 
tensorflow-1.0.0   tf\_gpu-1.0.0    2.7, 3.3-3.6   GCC 4.8     Bazel 0.4.2    5.1    8 
================== =============== ============== =========== ============== ====== ======

================= ============= ============== =================== ============= ===== ====
 *macOS*                          CPU (& if GPU)                                   GPU
------------------------------- ------------------------------------------------ ----------
 CPU Version       GPU Version  Python Version  Compiler            Build Tool   cuDNN CUDA
================= ============= ============== =================== ============= ===== ====
tensorflow-2.1.0  \              2.7, 3.5-3.7  Clang in Xcode 10.1  Bazel 0.27.1 
tensorflow-2.0.0  \              2.7, 3.3-3.7  Clang in Xcode 10.1  Bazel 0.26.1 
tensorflow-1.14.0 \              2.7, 3.3-3.7   Clang in Xcode      Bazel 0.24.1 
tensorflow-1.13.1 \              2.7, 3.3-3.7   Clang in Xcode      Bazel 0.19.2 
tensorflow-1.12.0 \              2.7, 3.3-3.6   Clang in Xcode      Bazel 0.15.0 
tensorflow-1.11.0 \              2.7, 3.3-3.6   Clang in Xcode      Bazel 0.15.0 
tensorflow-1.10.0 \              2.7, 3.3-3.6   Clang in Xcode      Bazel 0.15.0 
tensorflow-1.9.0  \              2.7, 3.3-3.6   Clang in Xcode      Bazel 0.11.0 
tensorflow-1.8.0  \              2.7, 3.3-3.6   Clang in Xcode      Bazel 0.10.1 
tensorflow-1.7.0  \              2.7, 3.3-3.6   Clang in Xcode      Bazel 0.10.1 
tensorflow-1.6.0  \              2.7, 3.3-3.6   Clang in Xcode      Bazel 0.8.1  
tensorflow-1.5.0  \              2.7, 3.3-3.6   Clang in Xcode      Bazel 0.8.1  
tensorflow-1.4.0  \              2.7, 3.3-3.6   Clang in Xcode      Bazel 0.5.4  
tensorflow-1.3.0  \              2.7, 3.3-3.6   Clang in Xcode      Bazel 0.4.5  
tensorflow-1.2.0  \              2.7, 3.3-3.6   Clang in Xcode      Bazel 0.4.5  
tensorflow-1.1.0  tf_gpu-1.1.0   2.7, 3.3-3.6   Clang in Xcode      Bazel 0.4.2   5.1   8 
tensorflow-1.0.0  tf_gpu-1.0.0   2.7, 3.3-3.6   Clang in Xcode      Bazel 0.4.2   5.1   8 
================= ============= ============== =================== ============= ===== ====




-------------------------
PyTorch
-------------------------





