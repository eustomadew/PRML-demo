==============
Installation
==============


------------------------
NVIDIA CUDA+cuDNN
------------------------

cuDNN  

- Download: https://developer.nvidia.com/rdp/cudnn-download 
- Archive: https://developer.nvidia.com/rdp/cudnn-archive 

=========== ==== ==== ==== ==== === === === === === === === ================
*NVIDIA*                    CUDA                             Notice: 
----------- ----------------------------------------------- ----------------
 cuDNN      11.0 10.2 10.1 10.0 9.2 9.1 9.0 8.0 7.5 7.0 6.5  Date
=========== ==== ==== ==== ==== === === === === === === === ================
 v8.0.2       √    √    √                                      Jul 24, 2020 
 v8.0.1 RC2   √    √                                           Jun 26, 2020 
 v7.6.5            √    √    √   √       √                   Nov 18/5, 2019 
 v7.6.4                 √    √   √       √                     Sep 27, 2019 
 v7.6.3                 √    √   √       √                     Aug 23, 2019 
 v7.6.2                 √    √   √       √                     Jul 22, 2019 
 v7.6.1                 √    √   √       √                     Jun 24, 2019 
 v7.6.0                 √    √   √       √                     May 20, 2019 
 v7.5.1                 √    √   √       √                     Apr 22, 2019 
 v7.5.0                 √    √   √       √                  Feb 25/21, 2019 
 v7.4.2                      √   √       √                     Dec 14, 2018 
 v7.4.1                      √   √       √                      Nov 8, 2018 
 v7.3.1                      √   √       √                     Sep 28, 2018 
 v7.3.0                      √           √                     Sep 19, 2018 
 v7.2.1                          √                              Aug 7, 2018 
 v7.1.4                          √       √   √                 May 16, 2018 
 v7.1.3                              √   √   √                 Apr 17, 2018 
 v7.1.2                          √   √   √                     Mar 21, 2018 
 v7.0.5                              √   √   √               Dec 11/5, 2017 
 v7.0.4                                  √                     Nov 13, 2017 
 v6.0                                        √   √             Apr 27, 2017 
 v5.1                                        √   √             Jan 20, 2017 
 v5                                          √   √             May 27, 2016 
 v4                                                  √         Feb 10, 2016 
 v3                                                  √          Sep 8, 2015 
 v2                                                      √     Mar 17, 2015 
 v1                                                      √   (cuDNN 6.5 R1) 
=========== ==== ==== ==== ==== === === === === === === === ================


------------------
NVIDIA TensorRT
------------------

TensorRT 

- Intro: https://developer.nvidia.com/zh-cn/tensorrt
- Download: https://developer.nvidia.com/nvidia-tensorrt-download


TensorRT for Linux

============= ====================================== ================= =============
*NVIDIA*       CUDA                                   Ubuntu           CentOS/RedHat
------------- -------------------------------------- ----------------- -------------
TensorRT            11.0 10.2                              18.04 16.04 14.04 7 
============= ==== ====
7.1 GA (.3.4)   √    √ 
7.0         

============= ======================================== ============


TensorRT for Windows
======== 
*NVIDIA* 
-------- 
TensorRT 
======== 
RT 7.1





-------------------------
TensorFlow vs. PyTorch
-------------------------




=====  ======  ======
GPU with CUDA  Output
-------------  ------
  A      B     A or B
=====  ======  ======
False  False   False
True   False   True
False  True    True
True   True    True
=====  ======  ======


