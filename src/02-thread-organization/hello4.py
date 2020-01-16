import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule(r"""
__global__ void hello_from_gpu(void)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("Hello World from block %d and thread %d!\n", bid, tid);
}
""")

hello_from_gpu = mod.get_function("hello_from_gpu")
hello_from_gpu(grid=(2,1,1), block=(4,1,1))
