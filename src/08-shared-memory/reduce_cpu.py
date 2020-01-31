import numpy, math, sys, time

if len(sys.argv)>2 and sys.argv[1]=='-double':
    real_py = 'float64' 
else:
    real_py = 'float32'

NUM_REPEATS = 10
s2ms = 1000
N = 100000000
x = numpy.full((N,1), 1.23, dtype=real_py)

t_sum = 0
t2_sum = 0
for repeat in range(NUM_REPEATS+1):
    start = time.time()

    v_sum = x.sum()

    elapsed_time = (time.time()-start)*s2ms
    print("Time = {:.6f} ms.".format(elapsed_time))
    if repeat > 0:
        t_sum += elapsed_time
        t2_sum += elapsed_time * elapsed_time
t_ave = t_sum / NUM_REPEATS
t_err = math.sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave)
print("Time = {:.6f} +- {:.6f} ms.".format(t_ave, t_err))

print("sum = ", v_sum)