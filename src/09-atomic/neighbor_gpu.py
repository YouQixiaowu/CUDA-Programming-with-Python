import numpy, math, sys, re, os
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
void __global__ find_neighbor_atomic
(int *d_NN, int *d_NL, const real *d_x, const real *d_y, 
const int N, const int MN, const real cutoff_square)
{
    const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        d_NN[n1] = 0;
        const real x1 = d_x[n1];
        const real y1 = d_y[n1];
        for (int n2 = n1 + 1; n2 < N; ++n2)
        {
            const real x12 = d_x[n2] - x1;
            const real y12 = d_y[n2] - y1;
            const real distance_square = x12 * x12 + y12 * y12;
            if (distance_square < cutoff_square)
            {
                d_NL[n1 * MN + atomicAdd(&d_NN[n1], 1)] = n2;
                d_NL[n2 * MN + atomicAdd(&d_NN[n2], 1)] = n1;
            }
        }
    }
}

void __global__ find_neighbor_no_atomic
(int *d_NN, int *d_NL, const real *d_x, const real *d_y, 
const int N, const real cutoff_square)
{
    const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 < N)
    {
        int count = 0;
        const real x1 = d_x[n1];
        const real y1 = d_y[n1];
        for (int n2 = 0; n2 < N; ++n2)
        {
            const real x12 = d_x[n2] - x1;
            const real y12 = d_y[n2] - y1;
            const real distance_square = x12 * x12 + y12 * y12;
            if ((distance_square < cutoff_square) && (n2 != n1))
            {
                d_NL[(count++) * N + n1] = n2;
            }
        }
        d_NN[n1] = count;
    }
}""".replace('real', real_cpp))
find_neighbor_atomic = mod.get_function("find_neighbor_atomic")
find_neighbor_no_atomic = mod.get_function("find_neighbor_no_atomic")

def timing(d_NN, d_NL, d_x, d_y, N, MN, atomic):
    cutoff = 1.9
    cutoff_square = cutoff * cutoff
    NUM_REPEATS = 10
    t_sum = 0
    t2_sum = 0
    for repeat in range(NUM_REPEATS+1):
        start = drv.Event()
        stop = drv.Event()
        start.record()
        if atomic:
            find_neighbor_atomic(d_NN, d_NL, d_x, d_y, numpy.int32(N),
            numpy.int32(MN), numpy.__dict__[real_py](cutoff_square),
            grid=((N-1)//128+1, 1), 
            block=(128,1,1))
        else:
            find_neighbor_no_atomic(d_NN, d_NL, d_x, d_y, numpy.int32(N),
            numpy.__dict__[real_py](cutoff_square),
            grid=((N-1)//128+1, 1), 
            block=(128,1,1))
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

    drv.memcpy_dtoh(h_NN, d_NN)
    drv.memcpy_dtoh(h_NL, d_NL)
    NL = h_NL.tolist()
    NN = h_NN.tolist()
    if atomic:
        with open(os.path.join(sys.path[0], 'neighbor_atomic.txt' if atomic else 'neighbor.txt') , 'w') as f:
            for i in range(N):
                f.write('{0}\t{1}\n'.format(NN[i][0], '\t'.join([str(NL[i*MN+ii][0]) if ii<NN[i][0] else 'nan' for ii in range(MN)])))



MN = 10
x = []
y = []
with open(os.path.join(sys.path[0], 'xy.txt'),'r') as f:
    content = f.readlines()
    for coor in content:
        a = re.findall(r'((([1-9]\d*)|0)(\.\d+)?)', coor)
        try:
            x.append(float(a[0][0]))
            y.append(float(a[1][0]))
        except IndexError:
            pass
h_x = numpy.array(x, dtype=real_py)
h_y = numpy.array(y, dtype=real_py)
N = h_x.size
h_NN = numpy.zeros((N,1), dtype=numpy.int32)
h_NL = numpy.empty((N*MN,1), dtype=numpy.int32)

d_x = drv.mem_alloc(h_x.nbytes)
d_y = drv.mem_alloc(h_y.nbytes)
d_NN = drv.mem_alloc(h_NN.nbytes)
d_NL = drv.mem_alloc(h_NL.nbytes)
drv.memcpy_htod(d_x, h_x)
drv.memcpy_htod(d_y, h_y)

print("\nnot using atomicAdd:")
timing(d_NN, d_NL, d_x, d_y, N, MN, False)
print("\nusing atomicAdd:")
timing(d_NN, d_NL, d_x, d_y, N, MN, True)


