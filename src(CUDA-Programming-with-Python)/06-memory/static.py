import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy


mod = SourceModule(r"""
__device__ int x = 1;
__device__ int y[2];

void __global__ my_kernel(void)
{
    y[0] = x + 1;
    y[1] = x + 2;
    printf("x = %d, y[0] = %d, y[1] = %d.\n", x, y[0], y[1]);
}
""")

my_kernel = mod.get_function("my_kernel")
my_kernel(grid=(1, 1), block=(1, 1, 1))
