import pycuda.autoinit
import pycuda.driver as drv
import numpy, math, sys
from pycuda.compiler import DynamicSourceModule

if len(sys.argv)==2 and sys.argv[1]=='-double':
    real_py = 'float64' 
    real_cpp = 'double'
else:
    real_py = 'float32'
    real_cpp = 'float'

mod = DynamicSourceModule(r"""
void __global__ add(const real *x, const real *y, real *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N) 
    {
        z[n] = x[n] + y[n];
    }
}
""".replace('real', real_cpp))
add = mod.get_function("add")

EPSILON = 1e-15
NUM_REPEATS = 10
a = 1.23
b = 2.34
c = 3.57
N = 100000000
h_x = numpy.full((N,1), a, dtype=real_py)
h_y = numpy.full((N,1), b, dtype=real_py)
h_z = numpy.zeros_like(h_x, dtype=real_py)
d_x = drv.mem_alloc(h_x.nbytes)
d_y = drv.mem_alloc(h_y.nbytes)
d_z = drv.mem_alloc(h_z.nbytes)
drv.memcpy_htod(d_x, h_x)
drv.memcpy_htod(d_y, h_y)
t_sum = 0
t2_sum = 0
for repeat in range(NUM_REPEATS+1):
    start = drv.Event()
    stop = drv.Event()
    start.record()
    
    add(d_x, d_y, d_z, numpy.int32(N), grid=((N-1)//128+1, 1), block=(128,1,1))

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

drv.memcpy_dtoh(h_z, d_z)
print('No errors' if (abs(h_z-c)<EPSILON).all() else 'Has errors')
