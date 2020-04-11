import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
array = gpuarray.to_gpu(np.array([[0,1,2], [3,4,5]], dtype='double'))

print('array:\n{0}'.format(array))
print('array+2:\n{0}'.format(array+2))
print('array-2:\n{0}'.format(array-2))
print('array*2:\n{0}'.format(array*2))
print('array/2:\n{0}'.format(array/2))
print('array**2:\n{0}'.format(array**2))
