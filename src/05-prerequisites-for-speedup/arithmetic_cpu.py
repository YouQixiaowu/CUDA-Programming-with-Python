import pycuda.autoinit
import pycuda.driver as drv
import numpy, math, sys, time
from functools import partial

if len(sys.argv)==2 and sys.argv[1]=='-double':
    real_py = 'float64' 
else:
    real_py = 'float32'

def arithmetic(x, x0):
    while math.sqrt(x) < x0: 
        x+=1
    return x  
    
s2ms = 1000
NUM_REPEATS = 10
x0 = 100
t_sum = 0
t2_sum = 0
for repeat in range(NUM_REPEATS+1):
    x = numpy.zeros((NUM_REPEATS,1), dtype=real_py)
    start = time.time()

    x = list(map(partial(arithmetic, x0=x0), x))

    elapsed_time = (time.time()-start)*s2ms
    print("Time = {:.6f} ms.".format(elapsed_time))
    if repeat > 0:
        t_sum += elapsed_time
        t2_sum += elapsed_time * elapsed_time

t_ave = t_sum / NUM_REPEATS
t_err = math.sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave)
print("Time = {:.6f} +- {:.6f} ms.".format(t_ave, t_err))
