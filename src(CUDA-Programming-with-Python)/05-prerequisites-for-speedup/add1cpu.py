import numpy, math, sys, time

if len(sys.argv)==2 and sys.argv[1]=='-double':
    real_py = 'float64' 
    EPSILON = 1e-15
else:
    real_py = 'float32'
    EPSILON = 1e-6

NUM_REPEATS = 10
a = 1.23
b = 2.34
c = 3.57
N = 100000000
x = numpy.full((N,1), a, dtype=real_py)
y = numpy.full((N,1), b, dtype=real_py)
z = numpy.zeros_like(x, dtype=real_py)
t_sum = 0
t2_sum = 0
for repeat in range(NUM_REPEATS+1):
    start = time.time()

    z = x + y
    
    elapsed_time = (time.time()-start)*1000
    print("Time = {:.6f} ms.".format(elapsed_time))
    if repeat > 0:
        t_sum += elapsed_time
        t2_sum += elapsed_time * elapsed_time

t_ave = t_sum / NUM_REPEATS
t_err = math.sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave)
print("Time = {:.6f} +- {:.6f} ms.".format(t_ave, t_err))
print('No errors' if (abs(z-c)<EPSILON).all() else 'Has errors')