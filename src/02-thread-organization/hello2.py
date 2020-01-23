import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

kernel_code = r"""
__global__ void hello_from_gpu(void)
{
    printf("hello World from the GPU!\n");  
}
"""
mod = SourceModule(kernel_code)
hello_from_gpu = mod.get_function("hello_from_gpu")
hello_from_gpu(grid=(1,1,1), block=(1,1,1))
