import numpy, math, sys, time
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
void __global__ gpu_sum(const real *x, const real *y, real *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        z[n] = x[n] + y[n];
    }
}""".replace('real', real_cpp))
gpu_sum = mod.get_function("gpu_sum")

def cpu_sum(x, y, N_host):
    z = numpy.empty_like(x, dtype=real_py)
    for n in range(N_host):
        z[n] = x[n] + y[n]
    return z

def timing(h_x, h_y, h_z, d_x, d_y, d_z, ratio, overlap):
    NUM_REPEATS = 10
    N = h_x.size
    t_sum = 0
    t2_sum = 0
    for repeat in range(NUM_REPEATS+1):
        start = time.time()

        if not overlap:
            cpu_sum(h_x, h_y, N//ratio)
        
        gpu_sum(d_x, d_y, d_z, numpy.int32(N), 
            grid=((N-1)//128+1, 1), 
            block=(128,1,1)
            )

        if overlap:
            cpu_sum(h_x, h_y, N//ratio)
        
        elapsed_time = (time.time()-start)*1000
        print("Time = {:.6f} ms.".format(elapsed_time))
        if repeat > 0:
            t_sum += elapsed_time
            t2_sum += elapsed_time * elapsed_time
    t_ave = t_sum / NUM_REPEATS
    t_err = math.sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave)
    print("Time = {:.6f} +- {:.6f} ms.".format(t_ave, t_err))



N = 100000000
h_x = numpy.full((N,1), 1.23, dtype=real_py)
h_y = numpy.full((N,1), 2.34, dtype=real_py)
h_z = numpy.zeros_like(h_x, dtype=real_py)
d_x = drv.mem_alloc(h_x.nbytes)
d_y = drv.mem_alloc(h_y.nbytes)
d_z = drv.mem_alloc(h_z.nbytes)
drv.memcpy_htod(d_x, h_x)
drv.memcpy_htod(d_y, h_y)

print("Without CPU-GPU overlap (ratio = 1000000)")
timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000000, False)
print("With CPU-GPU overlap (ratio = 1000000)")
timing(h_x, h_y, h_z, d_x, d_y, d_z, 1000000, True)

print("Without CPU-GPU overlap (ratio = 100000)")
timing(h_x, h_y, h_z, d_x, d_y, d_z, 100000, False)
print("With CPU-GPU overlap (ratio = 100000)")
timing(h_x, h_y, h_z, d_x, d_y, d_z, 100000, True)

print("Without CPU-GPU overlap (ratio = 10000000)")
timing(h_x, h_y, h_z, d_x, d_y, d_z, 10000000, False)
print("With CPU-GPU overlap (ratio = 10000000)")
timing(h_x, h_y, h_z, d_x, d_y, d_z, 10000000, True)
