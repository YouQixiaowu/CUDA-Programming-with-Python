import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy
num = 4
A = numpy.random.rand(num)
B = numpy.random.rand(num)
A_GPU = gpuarray.to_gpu(A.astype(numpy.float32))
B_GPU = gpuarray.to_gpu(B.astype(numpy.float32))
C_GPU = A_GPU + B_GPU
C = C_GPU.get()
print('A=', A)
print('B=', B)
print('C=', C)

