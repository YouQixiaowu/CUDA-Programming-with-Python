import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import numpy as np

dtype = np.double

h_array = np.array([[11,12,13],[21,22,23],[31,32,33],[41,42,43]], dtype=dtype)

a = gpuarray.to_gpu(h_array)
print('a:\n{0}\nshape={1}\n'.format(a.get(), a.shape))

stream = drv.Stream()
b = gpuarray.to_gpu_async(h_array, stream=stream)
print('b:\n{0}\nshape={1}\n'.format(b.get(), b.shape))

c = gpuarray.empty((100,100), dtype=dtype)
print('c:\n{0}\nshape={1}\n'.format(c, c.shape))

d = gpuarray.zeros((100,100), dtype=dtype)
print('d:\n{0}\nshape={1}\n'.format(d, d.shape))

e = gpuarray.arange(0.0,100.0,1.0,dtype=dtype)
print('e:\n{0}\nshape={1}\n'.format(e, e.shape))

f = gpuarray.if_positive(e<50, e-100, e+100)
print('f:\n{0}\nshape={1}\n'.format(f, f.shape))

g = gpuarray.if_positive(e<50, gpuarray.ones_like(e), gpuarray.zeros_like(e))
print('g:\n{0}\nshape={1}\n'.format(g, g.shape))

h = gpuarray.maximum(e, f)
print('h:\n{0}\nshape={1}\n'.format(h, h.shape))

i = gpuarray.minimum(e, f)
print('i:\n{0}\nshape={1}\n'.format(i, i.shape))

g = gpuarray.sum(a)
print(g, type(g))

k = gpuarray.max(a)
print(k, type(k))

l = gpuarray.min(a)
print(l, type(l))
