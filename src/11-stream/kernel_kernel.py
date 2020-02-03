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
void __global__ add(real *d_x, real *d_y, real *d_z, const int N1)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N1)
    {
        for (int i = 0; i < 100000; ++i)
        {
            d_z[n] = d_x[n] + d_y[n];
        }
    }
}""".replace('real', real_cpp))
add = mod.get_function("add")

global NUM_REPEATS
global N1 
global MAX_NUM_STREAMS
global N 
global block_size 
global streams
NUM_REPEATS = 10
N1 = 1024
MAX_NUM_STREAMS = 30
N = N1 * MAX_NUM_STREAMS
block_size = 128

def timing(d_x, d_y, d_z, num):
    t_sum = 0
    t2_sum = 0
    for repeat in range(NUM_REPEATS+1):
        start = drv.Event()
        stop = drv.Event()
        start.record() 

        for n in range(num):
            offset = n*N1
            add(
                numpy.int64(int(d_x)+offset), 
                numpy.int64(int(d_y)+offset), 
                numpy.int64(int(d_z)+offset), 
                numpy.int32(N1), 
                grid=((N1-1)//block_size+1, 1), 
                block=(128,1,1), 
                stream=streams[n]
                )

        stop.record()
        stop.synchronize()
        elapsed_time = start.time_till(stop)
        if repeat > 0:
            t_sum += elapsed_time
            t2_sum += elapsed_time * elapsed_time
    t_ave = t_sum / NUM_REPEATS
    t_err = math.sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave)
    print("Time = {:.6f} +- {:.6f} ms.".format(t_ave, t_err))


h_x = numpy.full((1,N), 1.23, dtype=real_py)
h_y = numpy.full((1,N), 2.34, dtype=real_py)
d_x = drv.mem_alloc(h_x.nbytes)
d_y = drv.mem_alloc(h_y.nbytes)
d_z = drv.mem_alloc(h_y.nbytes)
streams = []
for n in range(MAX_NUM_STREAMS):
    streams.append(drv.Stream())

for num in range(MAX_NUM_STREAMS):
    timing(d_x,d_y,d_z,num+1)
