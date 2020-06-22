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
num = 6
A = numpy.random.rand(num).astype(numpy.float32)
B = numpy.random.rand(num).astype(numpy.float32)
C = numpy.zeros(num).astype(numpy.float32)
A_GPU = drv.mem_alloc(A.nbytes)
B_GPU = drv.mem_alloc(B.nbytes)
C_GPU = drv.mem_alloc(C.nbytes)
drv.memcpy_htod(A_GPU, A)
drv.memcpy_htod(B_GPU, B)
add(A_GPU, B_GPU, C_GPU, grid=(2, 1), block=(4,1,1))
drv.memcpy_dtoh(C, C_GPU)
A_GPU.free()
B_GPU.free()
C_GPU.free()
print('A=', A)
print('B=', B)
print('C=', C)
