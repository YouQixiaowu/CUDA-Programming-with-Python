import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import DynamicSourceModule
import numpy as np
mod = DynamicSourceModule(r"""
void __global__ fun(const double *d_array)
{
    printf("%lf\n", d_array[0]);
}
""")
fun = mod.get_function('fun')
size_double = np.dtype('double').itemsize
h_array = np.array([1,2,3,4,5], dtype=np.double)
d_array = drv.mem_alloc(h_array.nbytes)
drv.memcpy_htod(d_array, h_array)
fun( np.uintp( int(d_array)+size_double*2 ), block=(1,1,1))
h_array2 = np.zeros((1,3), dtype=np.double)
drv.memcpy_dtoh(h_array2, int(d_array)+size_double)
print(h_array2)
