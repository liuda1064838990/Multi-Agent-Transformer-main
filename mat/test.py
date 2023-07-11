import numpy as np
dones_env=np.array([True,False])
b=dones_env == True
a=(dones_env == True).sum()
c=np.zeros(((dones_env == True).sum(), 3, 1), dtype=np.float32)
print(a)
d=[]
for i in range(5):
    d.append([1,2,3])
d[-1]=[2,3,4]
print(d)