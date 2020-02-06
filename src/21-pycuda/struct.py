import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
mod = SourceModule(r"""
struct st_data {
    int x;
    double y;
};
__global__ void function(const st_data data, st_data* datas, const int N) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n < N) 
    {
        datas[n].x = data.x;
        datas[n].y = data.y;
    }
}
""")
function = mod.get_function('function')
st_data = np.dtype({
        'names':['x','y'],
        'formats':[np.int32,np.double]}, align=True)
N=10
data = np.array((5, 3.5), dtype=st_data)
h_datas = np.zeros((N,), dtype=st_data)
d_datas = drv.mem_alloc(h_datas.nbytes)
print(h_datas)
drv.memcpy_htod(d_datas, h_datas)
function(data, d_datas, np.int32(N), block=(N,1,1))
drv.memcpy_dtoh(h_datas, d_datas)
print(h_datas)
