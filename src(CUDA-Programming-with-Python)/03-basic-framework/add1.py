import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy

mod = SourceModule(r"""
void __global__ add(const float *x, const float *y, float *z)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}
""")
add = mod.get_function("add")

EPSILON = 1e-15
a = 1.23
b = 2.34
c = 3.57
N = 100000000
h_x = numpy.full((N,1), a, dtype=numpy.float32)
h_y = numpy.full((N,1), b, dtype=numpy.float32)
h_z = numpy.zeros_like(h_x, dtype=numpy.float32)
d_x = drv.mem_alloc(h_x.nbytes)
d_y = drv.mem_alloc(h_y.nbytes)
d_z = drv.mem_alloc(h_z.nbytes)
drv.memcpy_htod(d_x, h_x)
drv.memcpy_htod(d_y, h_y)
add(d_x, d_y, d_z, grid=(N//128, 1), block=(128,1,1))
drv.memcpy_dtoh(h_z, d_z)
print('No errors' if (abs(h_z-c)<EPSILON).all() else 'Has errors')