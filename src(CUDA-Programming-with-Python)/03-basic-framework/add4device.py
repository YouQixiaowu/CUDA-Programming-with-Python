import pycuda.autoinit
import pycuda.driver as drv
import numpy
from pycuda.compiler import DynamicSourceModule

import numpy


mod1 = DynamicSourceModule(r"""
double __device__ add1_device(double x, double y)
{
    return (x + y);
}

void __global__ add1(double *x, double *y, double *z, int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N) 
    {
        z[n] = add1_device(x[n], y[n]);
    }
}
""")
add1 = mod1.get_function("add1")

mod2 = DynamicSourceModule(r"""
double __device__ add2_device(double x, double y, double *z)
{
    *z = x + y;
}

void __global__ add2(double *x, double *y, double *z, int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N) 
    {
        add2_device(x[n], y[n], &z[n]);
    }
}
""")
add2 = mod2.get_function("add2")

mod3 = DynamicSourceModule(r"""
double __device__ add3_device(double x, double y, double &z)
{
    z = x + y;
}

void __global__ add3(double *x, double *y, double *z, int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N) 
    {
        add3_device(x[n], y[n], z[n]);
    }
}
""")
add3 = mod3.get_function("add3")

EPSILON = 1e-15
a = 1.23
b = 2.34
c = 3.57
N = 100000001
h_x = numpy.full((N,1), a)
h_y = numpy.full((N,1), b)
h_z = numpy.zeros_like(h_x)
d_x = drv.mem_alloc(h_x.nbytes)
d_y = drv.mem_alloc(h_y.nbytes)
d_z = drv.mem_alloc(h_z.nbytes)
drv.memcpy_htod(d_x, h_x)
drv.memcpy_htod(d_y, h_y)

add1(d_x, d_y, d_z, numpy.int32(N), grid=((N-1)//128+1, 1), block=(128, 1, 1))
drv.memcpy_dtoh(h_z, d_z)
print('No errors' if (abs(h_z-c)<EPSILON).all() else 'Has errors')

add2(d_x, d_y, d_z, numpy.int32(N), grid=((N-1)//128+1, 1), block=(128, 1, 1))
drv.memcpy_dtoh(h_z, d_z)
print('No errors' if (abs(h_z-c)<EPSILON).all() else 'Has errors')

add3(d_x, d_y, d_z, numpy.int32(N), grid=((N-1)//128+1, 1), block=(128, 1, 1))
drv.memcpy_dtoh(h_z, d_z)
print('No errors' if (abs(h_z-c)<EPSILON).all() else 'Has errors')