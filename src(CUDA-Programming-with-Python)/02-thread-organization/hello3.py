import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule(r"""
__global__ void hello_from_gpu(void)
{
    printf("hello World from the GPU!\n");  
}
""")

hello_from_gpu = mod.get_function("hello_from_gpu")
hello_from_gpu(grid=(2,1,1),block=(4,1,1))
