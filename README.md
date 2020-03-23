# CUDA-Programming-for-Python
<<CUDA-Programming编程>>书籍内容将不再包含pycuda的内容，相关的pycuda的文字教程将以博客的形式给出。将在完成时给出链接。
这些代码原是为樊哲勇老师的书籍<<CUDA-Programming编程>>编写的示例代码。为了让CUDA初学者在python中更好的使用CUDA，本代码库应用了[pycuda](https://mathema.tician.de/software/pycuda/)模块，构建了与书中使用C++编写的功能上一致的例子。

## 运行环境要求
1. 适用CUDA的显卡
2. cuda 10.1，并已经配置环境变量
3. python3.7 64位
4. numpy模块  （安装方法:pip install numpy）
5. pycuda模块 （安装方法:pip install pycuda）
6. skcuda模块 （安装方法:pip install scikit-cuda）
7. 注意：尚未在linux进行测试（即将完成）

## 目录和源代码列表
本仓库中大部分代码与C++版的同名代码功能一致。具体详情请参考[CUDA-Programming](https://github.com/brucefan1983/CUDA-Programming/blob/master/README.md)当中README.md相关内容的介绍。
### 第1章
没有代码
### 第2与3章
详细说明参考C++版代码
### 第4章
在python中使用CUDA无需额外的宏来捕捉异常。
### 第5至11章
详细说明参考C++版代码
### 第12章
pycuda关于同一设备内存相关内容尚属于实验功能，故暂时不做介绍。
### 第13章
pycuda使用CUDA标准库可以使用[skcuda](https://scikit-cuda.readthedocs.io/en/latest/index.html)，该模块依赖pycuda实现了标准库的访问，具体使用方法不做介绍。
### 第14章
详细说明参考C++版代码
### 第15章
没有代码
### 第16至18章
由于使用Pycuda进行CPU端计算，效率相比C++进行CPU端计算速度缓慢约500倍。故暂时暂时不做介绍。
### 第19章
| 文件         | 内容            |
|:-------------|:----------------|
|Ar.py         |相当于main函数中的内容，它将调用其他自定义模块进行Ar的分子动力学模拟。|
|material.py   |Material类，原胞信息的检查与超晶胞操作。|
|md.py         |MolecularDynamics类，分子动力学模拟的准备工作与流程控制。
|GPU.py        |Kernel类，核函数的编译，内存的分配，提供函数接口，程序浮点数精度的控制。|
|Kernel.cu     |该文件非Python执行文件，CUDA的核函数均在此文件中，由Kernel类读取并编译为核函数。|
### 第20章
pycuda关于同一设备内存相关内容尚属于实验功能，故暂时不做介绍。
### 第21章
该章代码特别为说明pycuda的使用而编写，在C++版本的代码中找不到对应的代码。
| 文件          | 内容                  |
|:--------------|:----------------------|
| add1.py       |使用gpuarray实现矩阵相加|
| add2.py       |在核函数中使用gpuarray实现矩阵相加|
| pointer.py    |在pycuda中进行指针偏移|
| struct.py     |在核函数中使用结构体进行传参|
| random.py     |pycuda产生随机数数组|
| define.py     |在核函数中使用宏|
| function_0.py |GPUArray的基本运算|
| function_1.py |GPUArray的基础函数|
| function_2.py |gpuarray模块的一些函数|
| function_3.py |gpuarray模块的元素级的数学函数|
