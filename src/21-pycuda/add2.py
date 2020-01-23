import pycuda.gpuarray as gpuarray
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
d_x = gpuarray.to_gpu(numpy.full((N,1), a).astype(numpy.float32))
d_y = gpuarray.to_gpu(numpy.full((N,1), b).astype(numpy.float32))
d_z = gpuarray.to_gpu(numpy.zeros((N,1)).astype(numpy.float32))
add(d_x, d_y, d_z, grid=(N//128, 1), block=(128,1,1))
h_z = d_z.get()
print('No errors' if (abs(h_z-c)<EPSILON).all() else 'Has errors')