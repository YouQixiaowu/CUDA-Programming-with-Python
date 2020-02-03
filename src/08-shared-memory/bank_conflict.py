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
    N = 20000



mod = DynamicSourceModule(r'''
const int TILE_DIM = 32;
__global__ void transpose1(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 < N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

__global__ void transpose2(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM + 1];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 < N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}'''.replace('real', real_cpp))

transpose1 = mod.get_function("transpose1")
transpose2 = mod.get_function("transpose2")



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
        if task == 1:
            transpose1(d_A, d_B, numpy.int32(N), 
                grid=grid_size, 
                block=block_size
                )
        elif task == 2:
            transpose2(d_A, d_B, numpy.int32(N), 
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

N2 = N**2
h_A = numpy.arange(0, N2, 1, dtype=real_py)
h_B = numpy.zeros_like(h_A, dtype=real_py)
d_A = drv.mem_alloc(h_A.nbytes)
d_B = drv.mem_alloc(h_B.nbytes)
drv.memcpy_htod(d_A, h_A)
print("\ntranspose with shared memory bank conflict:")
timing(d_A, d_B, N, 1)
print("\ntranspose without shared memory bank conflict:\n")
timing(d_A, d_B, N, 2)

drv.memcpy_dtoh(h_B, d_B)

if N<=10:
    print("A =")
    print(h_A.reshape(N,N))
    print("B =")
    print(h_B.reshape(N,N))






