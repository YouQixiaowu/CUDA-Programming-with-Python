import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.curandom as curandom
a = curandom.rand((5,3))
print('a:\n{0}\n{1}\n{2}\n'.format(a, a.dtype, type(a)))
b = curandom.seed_getter_uniform(5)
print('seed_getter_uniform:\n{0}\n{1}\n'.format(b, b.dtype))
c = curandom.seed_getter_unique(5)
print('seed_getter_unique:\n{0}\n{1}\n'.format(c, c.dtype))
generator = curandom.XORWOWRandomNumberGenerator(curandom.seed_getter_unique, 1000
d = gpuarray.empty((5,3), dtype = 'float32')
generator.fill_uniform(d)
print('d:\n{0}\n{1}\n{2}\n'.format(d, d.dtype, type(d)))
e = generator.gen_uniform((5,3), dtype = 'float32')
print('e:\n{0}\n{1}\n{2}\n'.format(e, e.dtype, type(e)))



