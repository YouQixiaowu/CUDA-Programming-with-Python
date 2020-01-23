import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

EPSILON = 1e-15
a = 1.23
b = 2.34
c = 3.57
N = 100000000
d_x = gpuarray.to_gpu(numpy.full((N,1), a).astype(numpy.float32))
d_y = gpuarray.to_gpu(numpy.full((N,1), b).astype(numpy.float32))
h_z = (d_x+d_y).get()
print('No errors' if (abs(h_z-c)<EPSILON).all() else 'Has errors')