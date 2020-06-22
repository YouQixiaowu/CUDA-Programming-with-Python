import pycuda.autoinit
from pycuda.compiler import SourceModule

kernel_code = r"""
__global__ void print_id(void)
{
    printf("blockIdx.x = %d; threadIdx.x = %d;\n", blockIdx.x, threadIdx.x);
}
"""
mod = SourceModule(kernel_code)
print_id = mod.get_function("print_id")
print_id(grid=(2,), block=(3,1,1))