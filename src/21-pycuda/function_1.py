import pycuda.gpuarray as ga
import pycuda.autoinit
import numpy as np

dtype = np.double

array = ga.to_gpu(np.array([[11,12,13],[21,22,23],[31,32,33],[41,42,43]], dtype=dtype))

print(array.shape)
print(array.size)
print(array.dtype)
print(array.nbytes)
print(array.ptr)

print('array:\n{0}\nshape={1}\n'.format(array, array.shape))
print('array:\n{0}\nshape={1}\n'.format(array.get(), array.shape))

b = array.reshape((2,6))
print('b:\n{0}\nshape={1}\n'.format(b.get(), b.shape))

c = array.ravel()
print('c:\n{0}\nshape={1}\n'.format(c.get(), c.shape))

d = array.T
print('d:\n{0}\nshape={1}\n'.format(d.get(), d.shape))

e = array.copy()
print('e:\n{0}\nshape={1}\n'.format(e.get(), e.shape))

f = array.astype(np.int32)
print('f:\n{0}\nshape={1}\n'.format(f.get(), f.shape))


