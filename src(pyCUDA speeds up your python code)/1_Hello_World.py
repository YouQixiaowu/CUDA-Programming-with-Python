import pycuda.autoinit
from pycuda.compiler import SourceModule

kernel_code = r"""
__global__ void hello_from_gpu(void)
{
    printf("Hello World from the GPU!\n");  
}
"""
mod = SourceModule(kernel_code)
hello_from_gpu = mod.get_function("hello_from_gpu")
hello_from_gpu(block=(3,4,5))
