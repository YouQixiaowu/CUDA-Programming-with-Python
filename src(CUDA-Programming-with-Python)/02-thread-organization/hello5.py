import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule(r"""
__global__ void hello_from_gpu(void)
{
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("Hello World from block-%d and thread-(%d, %d)!\n", b, tx, ty);
}
""")

hello_from_gpu = mod.get_function("hello_from_gpu")
hello_from_gpu(grid=(1,1,1), block=(2,4,1))
