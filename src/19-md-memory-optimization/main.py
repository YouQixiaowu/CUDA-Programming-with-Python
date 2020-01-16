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

mod = DynamicSourceModule(r'''
struct Box
{
    real lx;
    real ly;
    real lz;
    real lx2;
    real ly2;
    real lz2;
};
static void __global__ gpu_find_neighbor
(
    int N, int MN, int *g_NN, int *g_NL, Box box, 
    real *g_x, real *g_y, real *g_z, real cutoff2
)
{
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        int count = 0;
        real x1 = g_x[n1];
        real y1 = g_y[n1];
        real z1 = g_z[n1];
        for (int n2 = 0; n2 < N; n2++)
        {
            real x12 = g_x[n2] - x1;
            real y12 = g_y[n2] - y1;
            real z12 = g_z[n2] - z1;
            apply_mic(box, &x12, &y12, &z12);
            real d12_square = x12*x12 + y12*y12 + z12*z12;
            if ((n2 != n1) && (d12_square < cutoff2))
            {
                g_NL[count++ * N + n1] = n2;
            }
        }
        g_NN[n1] = count;
    }
}
'''.replace('real', real_cpp))
gpu_find_neighbor = mod.get_function("gpu_find_neighbor")


class MD(object):
    def __init__(self, double=False):
        if double:
            real_py = 'float64' 
            real_cpp = 'double'
        else:
            real_py = 'float32'
            real_cpp = 'float'
        kB = 8.617343e-5
        natural2fs = 1.018051e+1
        nx = 4              # 超胞数
        Ne = 2000           # 平衡阶段步数
        Np = 2000           # 产出阶段步数
        N = 4*nx*nx*nx      # 原子总数
        Ns = 100            # 产出间隔
        MN = 200            # 每个原子最多的邻居原子数
        T_0 = 60.0
        ax = 5.385
        time_step = 5.0 / natural2fs
        # 质量
        m = numpy.full((N,1), 40, dtype=real_py)
        # 坐标
        box = numpy.array([ax*nx]*3+[ax*nx*0.5]*3)
        x0 = [0.0, 0.0, 0.5, 0.5]
        y0 = [0.0, 0.5, 0.0, 0.5]
        z0 = [0.0, 0.5, 0.5, 0.0]
        temp_x = []
        temp_y = []
        temp_z = []
        for ix in range(nx):
            for iy in range(nx):
                for iz in range(nx):
                    for i in range(4):
                        temp_x.append((ix + x0[i]) * ax)
                        temp_y.append((iy + y0[i]) * ax)
                        temp_z.append((iz + z0[i]) * ax)
        x = numpy.array(temp_x)
        y = numpy.array(temp_y)
        z = numpy.array(temp_z)
        # 速度
        vx = -1 + 2*numpy.random.rand(N).astype(real_py)
        vy = -1 + 2*numpy.random.rand(N).astype(real_py)
        vz = -1 + 2*numpy.random.rand(N).astype(real_py)
        vx = vx - numpy.sum(vx*m) / m
        vy = vy - numpy.sum(vy*m) / m
        vz = vz - numpy.sum(vz*m) / m

        temperature = numpy.sum(m*(vx*vx+vy*vy+vz*vz))/(3*kB*N)
        scale_factor = math.sqrt(T_0 / temperature)

        vx = vx * scale_factor
        vy = vy * scale_factor
        vz = vz * scale_factor

        g_m = drv.mem_alloc(m.nbytes)
        g_x = drv.mem_alloc(x.nbytes)
        g_y = drv.mem_alloc(y.nbytes)
        g_z = drv.mem_alloc(z.nbytes)

        g_vx = drv.mem_alloc(vx.nbytes)
        g_vy = drv.mem_alloc(vy.nbytes)
        g_vz = drv.mem_alloc(vz.nbytes)
        NN = numpy.full((N,1), 0, dtype=numpy.int32)
        g_NN = drv.mem_alloc(NN.nbytes)
        NL = numpy.full((N*MN,1), 0, dtype=numpy.int32)
        g_NL = drv.mem_alloc(NL.nbytes)

        g_box = drv.mem_alloc(box.nbytes)
        drv.memcpy_htod(g_m, m)
        drv.memcpy_htod(g_x, x)
        drv.memcpy_htod(g_y, y)
        drv.memcpy_htod(g_z, z)
        drv.memcpy_htod(g_vx, vx)
        drv.memcpy_htod(g_vy, vy)
        drv.memcpy_htod(g_vz, vz)
        drv.memcpy_htod(g_box, box)


        cutoff = 11.0
        cutoff2 = cutoff*cutoff
        block_size = 128
        grid_size = (N - 1) // block_size + 1
        gpu_find_neighbor(N, MN, g_NN, g_NL, g_box, g_x, g_y, g_z, cutoff2, grid = grid_size, block=(128,1,1))







md = MD()
