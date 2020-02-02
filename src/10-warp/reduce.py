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
const unsigned FULL_MASK = 0xffffffff;

extern "C"{void __global__ reduce_syncwarp(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncwarp();
    }

    if (tid == 0)
    {
        atomicAdd(d_y, s_y[0]);
    }
}}

extern "C"{void __global__ reduce_shfl(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];

    for (int offset = 16; offset > 0; offset >>= 1)
    {
        y += __shfl_down_sync(FULL_MASK, y, offset);
    }

    if (tid == 0)
    {
        atomicAdd(d_y, y);
    }
}
}

extern "C"{void __global__ reduce_cp(const real *d_x, real *d_y, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    extern __shared__ real s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    real y = s_y[tid];

    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
    for (int i = g.size() >> 1; i > 0; i >>= 1)
    {
        y += g.shfl_down(y, i);
    }

    if (tid == 0)
    {
        atomicAdd(d_y, y);
    }
}
}
""".replace('real', real_cpp), no_extern_c=True)
reduce_syncwarp = mod.get_function("reduce_syncwarp")
reduce_shfl = mod.get_function("reduce_shfl")
reduce_cp = mod.get_function("reduce_cp")



def timing(method):
    NUM_REPEATS = 10
    N = 100000000
    BLOCK_SIZE = 128
    grid_size = (N-1)//128+1
    h_x = numpy.full((N,1), 1.23, dtype=real_py)
    d_x = drv.mem_alloc(h_x.nbytes)
    drv.memcpy_htod(d_x, h_x)
    t_sum = 0
    t2_sum = 0
    for repeat in range(NUM_REPEATS+1):
        start = drv.Event()
        stop = drv.Event()
        start.record() 

        h_y = numpy.zeros((1,1), dtype=real_py)
        d_y = drv.mem_alloc(h_y.nbytes)
        drv.memcpy_htod(d_y, h_y)
        if method==0:
            reduce_syncwarp(d_x, d_y, numpy.int32(N), grid=(grid_size, 1), block=(128,1,1), shared=numpy.zeros((1,1),dtype=real_py).nbytes*BLOCK_SIZE)
        elif method==1:
            reduce_shfl(d_x, d_y, numpy.int32(N), grid=((N-1)//128+1, 1), block=(128,1,1), shared=numpy.zeros((1,1),dtype=real_py).nbytes*BLOCK_SIZE)
        elif method==2:
            reduce_cp(d_x, d_y, numpy.int32(N), grid=((N-1)//128+1, 1), block=(128,1,1), shared=numpy.zeros((1,1),dtype=real_py).nbytes*BLOCK_SIZE)
        else:
            print("Error: wrong method")
            break
        drv.memcpy_dtoh(h_y, d_y)
        v_sum = h_y[0,0]

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
    

print("\nusing syncwarp:")
timing(0)
print("\nusing shfl:")
timing(1)
print("\nusing cooperative group:")
timing(2)