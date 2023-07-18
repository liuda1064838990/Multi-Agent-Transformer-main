import numpy as np
dones_env=np.array([True,False])
a=[1,2,3]
b=a
c=a[:]
a[0]=0
print(b)
print(c)
