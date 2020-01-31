import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import DynamicSourceModule
import numpy as np

a = gpuarray.to_gpu(np.array([1,2,3,4],dtype=np.float32))
b = gpuarray.to_gpu(np.array([2,3,4,5],dtype=np.float32))
a*=b
print(a.get())
