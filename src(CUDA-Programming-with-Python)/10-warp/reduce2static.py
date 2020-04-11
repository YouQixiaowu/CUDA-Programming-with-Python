import pycuda.autoinit
import pycuda.driver as drv
import numpy, math, sys
from pycuda.compiler import DynamicSourceModule

if len(sys.argv)>2 and sys.argv[1]=='-double':
    real_py = 'float64' 
    real_cpp = 'double'
else:
    real_py = 'float32'
    real_cpp = 'float'

mod = DynamicSourceModule(r"""
#include <cooperative_groups.h>
using namespace cooperative_groups;

extern "C"{
void __global__ reduce_cp(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    extern __shared__ real s_y[];

    real y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (int n = bid * blockDim.x + tid; n < N; n += stride)
    {
        y += d_x[n];
    }
    s_y[tid] = y;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        y += g.shfl_down(y, i);
    }

    if (tid == 0)
    {
        d_y[bid] = y;
    }
}
}
""".replace('real', real_cpp), no_extern_c=True)
reduce_cp = mod.get_function("reduce_cp")

NUM_REPEATS = 10
N = 100000000
BLOCK_SIZE = 128
NUM_ROUNDS = 10

h_x = numpy.full((N,1), 1.23, dtype=real_py)
d_x = drv.mem_alloc(h_x.nbytes)
drv.memcpy_htod(d_x, h_x)

grid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
grid_size = (grid_size + NUM_ROUNDS - 1) // NUM_ROUNDS
size_real = numpy.dtype(real_py).itemsize

d_y = drv.mem_alloc(size_real*grid_size)

t_sum = 0
t2_sum = 0
for repeat in range(NUM_REPEATS+1):
    start = drv.Event()
    stop = drv.Event()
    start.record() 

    reduce_cp(d_x, d_y, numpy.int32(N), 
        block=(BLOCK_SIZE,1,1), 
        grid=(grid_size,1),
        shared=size_real*BLOCK_SIZE)

    reduce_cp(d_x, d_y, numpy.int32(N), 
        block=(1024,1,1), 
        grid=(1,1),
        shared=size_real*1024)

    h_y = numpy.array([0], dtype=real_py)
    drv.memcpy_dtoh(h_y, d_y)
    v_sum = h_y[0]

    stop.record()
    stop.synchronize()
    elapsed_time = start.time_till(stop)
    print("Time = {:.6f} ms.".format(elapsed_time))
    if repeat > 0:
        t_sum += elapsed_time
        t2_sum += elapsed_time * elapsed_time
t_ave = t_sum / NUM_REPEATS
t_err = math.sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave)
print("Time = {:.6f} +- {:.6f} ms.".format(t_ave, t_err))
print("sum = ", v_sum)

