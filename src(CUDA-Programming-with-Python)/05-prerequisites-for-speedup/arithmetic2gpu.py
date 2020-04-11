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

if len(sys.argv)==3:
    N = int(sys.argv[2])
else:
    N = 1000000

print('Type:{}; Number:{};'.format(real_cpp, N))

mod = DynamicSourceModule(r"""
void __global__ arithmetic(real *d_x, const real x0, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        real x_tmp = d_x[n];
        while (sqrt(x_tmp) < x0)
        {
            ++x_tmp;
        }
        d_x[n] = x_tmp;
    }
}""".replace('real', real_cpp))
arithmetic = mod.get_function("arithmetic")

NUM_REPEATS = 10
x0 = numpy.__dict__[real_py](100)

h_x = numpy.zeros((N,1), dtype=real_py)
d_x = drv.mem_alloc(h_x.nbytes)

t_sum = 0
t2_sum = 0
for repeat in range(NUM_REPEATS+1):
    drv.memcpy_htod(d_x, h_x)
    start = drv.Event()
    stop = drv.Event()
    start.record()

    arithmetic(d_x, x0, numpy.int32(N), 
        grid=((N-1)//128+1, 1), 
        block=(128,1,1)
        )
    
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
