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
    N = 10

mod = DynamicSourceModule(r'''
__global__ void copy(const real *A, real *B, const int N, const int TILE_DIM)
{
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    const int index = ny * N + nx;
    if (nx < N && ny < N)
    {
        B[index] = A[index];
    }
}

__global__ void transpose1(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx];
    }
}

__global__ void transpose2(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny];
    }
}

__global__ void transpose3(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = __ldg(&A[nx * N + ny]);
    }
}'''.replace('real', real_cpp))

copy = mod.get_function("copy")
transpose1 = mod.get_function("transpose1")
transpose2 = mod.get_function("transpose2")
transpose3 = mod.get_function("transpose3")

def timing(d_A, d_B, N, task):
    NUM_REPEATS = 10
    TILE_DIM = 32
    grid_size_x = (N + TILE_DIM - 1) // TILE_DIM
    grid_size_y = grid_size_x
    block_size = (TILE_DIM, TILE_DIM, 1)
    grid_size = (grid_size_x, grid_size_y, 1)

    t_sum = 0
    t2_sum = 0
    for repeat in range(NUM_REPEATS+1):
        start = drv.Event()
        stop = drv.Event()
        start.record()
        if task == 0:
            copy(d_A, d_B, numpy.int32(N), numpy.int32(TILE_DIM), 
                grid=grid_size, 
                block=block_size
                )
        elif task == 1:
            transpose1(d_A, d_B, numpy.int32(N), 
                grid=grid_size, 
                block=block_size
                )
        elif task == 2:
            transpose2(d_A, d_B, numpy.int32(N), 
                grid=grid_size, 
                block=block_size
                )
        elif task == 3:
            transpose3(d_A, d_B, numpy.int32(N), 
                grid=grid_size, 
                block=block_size
                )
        else:
            print("Error: wrong task")
            return
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


N2 = N*N
h_A = numpy.arange(0, N2, 1, dtype=real_py)
h_B = numpy.zeros_like(h_A, dtype=real_py)
d_A = drv.mem_alloc(h_A.nbytes)
d_B = drv.mem_alloc(h_B.nbytes)
drv.memcpy_htod(d_A, h_A)

print("copy:")
timing(d_A, d_B, N, 0)
print("transpose with coalesced read:")
timing(d_A, d_B, N, 1)
print("transpose with coalesced write:")
timing(d_A, d_B, N, 2)
print("transpose with coalesced write and __ldg read:")
timing(d_A, d_B, N, 3)

drv.memcpy_dtoh(h_B, d_B)
if N<=10:
    print("A =")
    print(h_A.reshape(N,N))
    print("B =")
    print(h_B.reshape(N,N))
