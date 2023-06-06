#!/usr/bin/env python
# coding: utf-8

# In[3]:


# create a graph with two cycles glued along a path
import numpy as np
tt = 60 # numpber of vertices in each cycle
m = 5 # number of shared vertices
n = 2 * tt - m # total number of vertices
P = np.zeros((n, n))

#fill first cycle
for ii in range(0, tt-1):
    P[ii, ii + 1] = 1

# last vertex in the first cycle splits
P[tt - 1, 0] = 0.5
P[tt - 1, tt] = 0.5

# second cycle
for ii in range(tt, n - 1):
    P[ii, ii + 1] = 1
    
# last vertex of second cycle joins back to the tt-m+1 vertex
P[n - 1, tt - m] = 1

# compute extended effective resistance on P
R = extEffRes(P)   
R = np.sqrt(R)

# compute the hitting probability matrix when beta=1/2
d12 = -np.log(get_Ahp(P))
for i in range(0, n):
    d12[i, i] = 0

# compute the hitting probability matrix when beta=1
d1 = -np.log(get_Ahp(P, 1))
for i in range(0, n):
    d1[i, i] = 0
assert (np.nanmax(((d1 - np.transpose(d1)) / d1)) < 0.001)
r = np.rint((tt - m) / 2)

