import numpy

EPSILON = 1e-15
a = 1.23
b = 2.34
c = 3.57
N = 100000000
x = numpy.full((N,1), a)
y = numpy.full((N,1), b)
z = x + y
print('No errors' if (abs(z-c)<EPSILON).all() else 'Has errors')
