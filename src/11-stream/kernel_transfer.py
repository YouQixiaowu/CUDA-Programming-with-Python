import numpy, math, sys
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import DynamicSourceModule

if len(sys.argv)>2 and sys.argv[1]=='-double':
    real_py = 'float64' 
    real_cpp = 'double'
else:
    real_py = 'float32'
    real_cpp = 'float'

mod = DynamicSourceModule(r"""
void __global__ add(const real *x, const real *y, real *z, int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        for (int i = 0; i < 40; ++i)
        {
            z[n] = x[n] + y[n];
        }
    }
}""".replace('real', real_cpp))
add = mod.get_function("add")

NUM_REPEATS = 10
N = 1 << 22
size_real = numpy.dtype(real_py).itemsize
M = size_real*N
MAX_NUM_STREAMS = 64
block_size = 128
h_x = numpy.full((1,N), 1.23, dtype=real_py)
h_y = numpy.full((1,N), 2.34, dtype=real_py)
h_z = numpy.zeros((1,N), dtype=real_py)
d_x = drv.mem_alloc(M)
d_y = drv.mem_alloc(M)
d_z = drv.mem_alloc(M)

streams = []
for n in range(MAX_NUM_STREAMS):
    streams.append(drv.Stream())

for c in range(7):
    num = 1<<c
    N1 = N//num
    t_sum = 0
    t2_sum = 0
    for repeat in range(NUM_REPEATS+1):
        start = drv.Event()
        stop = drv.Event()
        start.record() 

        for i in range(num):
            offset = i * N1
            drv.memcpy_htod_async(
                int(d_x)+offset, 
                h_x[0,offset:offset+N//num],
                stream=streams[i])

            drv.memcpy_htod_async(
                int(d_y)+offset, 
                h_y[0,offset:offset+N//num],
                stream=streams[i])

            add(
                numpy.uint64(int(d_x)+offset), 
                numpy.uint64(int(d_y)+offset), 
                numpy.uint64(int(d_z)+offset), 
                numpy.int32(N1), 
                grid=((N1-1)//block_size+1, 1), 
                block=(128,1,1), 
                stream=streams[i]
                )
            h_z_sub = h_z[0,offset:offset+N//num]
            drv.memcpy_dtoh_async(
                h_z_sub,
                int(d_z)+offset, 
                stream=streams[i])
            h_z[0,offset:offset+N//num] = h_z_sub

        stop.record()
        stop.synchronize()
        elapsed_time = start.time_till(stop)
        if repeat > 0:
            t_sum += elapsed_time
            t2_sum += elapsed_time * elapsed_time
    t_ave = t_sum / NUM_REPEATS
    t_err = math.sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave)
    print(num, t_ave)
