import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import numpy as np
import pycuda.cumath as cumath
dtype = np.double

h_array = np.array(
    [
        [-2,-1, 0],
        [ 1, 2, 3],
        [ 4, 5, 6],
        [ 7, 8, 9],
    ], 
    dtype=dtype
    )

array = gpuarray.to_gpu(h_array)

b = cumath.fabs(array)
print('b:\n{0}\nshape={1}\n'.format(b, b.shape))

c = cumath.exp(array)
print('c:\n{0}\nshape={1}\n'.format(c, c.shape))

d = cumath.sin(array)
print('d:\n{0}\nshape={1}\n'.format(d, d.shape))


