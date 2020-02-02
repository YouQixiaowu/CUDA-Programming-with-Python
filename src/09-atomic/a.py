
import numpy as np

a = np.random.rand(3,10)
a = a.tolist()
b = [3,4,2]
for i in range(len(b)):
    print('\t'.join(map(lambda x:str(x), a[i][0:b[i]])))