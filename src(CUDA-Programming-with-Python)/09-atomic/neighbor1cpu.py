import numpy, math, sys, re, time

if len(sys.argv)>2 and sys.argv[1]=='-double':
    real_py = 'float64' 
else:
    real_py = 'float32'

NUM_REPEATS = 10
MN = 10
cutoff = 1.9
cutoff_square = cutoff * cutoff

x = []
y = []
with open('src/09-atomic/xy2.txt','r') as f:
    content = f.readlines()
    for coor in content:
        a = re.findall(r'((([1-9]\d*)|0)(\.\d+)?)', coor)
        try:
            x.append(float(a[0][0]))
            y.append(float(a[1][0]))
        except IndexError:
            pass
x = numpy.array(x, dtype=real_py)
y = numpy.array(y, dtype=real_py)
N = x.size
t_sum = 0
t2_sum = 0
for repeat in range(NUM_REPEATS+1):
    start = time.time()
    NN = numpy.zeros((N,1), dtype=numpy.int32)
    NL = numpy.empty((N*MN,1), dtype=numpy.int32)
    for n1 in range(N):
        for n2 in range(n1+1, N):
            x12 = x[n2]-x[n1]
            y12 = y[n2]-y[n1]
            if (x12 * x12 + y12 * y12) < cutoff_square:
                NL[n1 * MN + NN[n1]] = n2
                NN[n1]+=1
                NL[n2 * MN + NN[n2]] = n1
                NN[n2]+=1

    elapsed_time = (time.time()-start)*1000
    print("Time = {:.6f} ms.".format(elapsed_time))
    if repeat > 0:
        t_sum += elapsed_time
        t2_sum += elapsed_time * elapsed_time
t_ave = t_sum / NUM_REPEATS
t_err = math.sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave)
print("Time = {:.6f} +- {:.6f} ms.".format(t_ave, t_err))

