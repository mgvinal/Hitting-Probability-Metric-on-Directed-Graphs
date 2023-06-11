#!/usr/bin/env python
# coding: utf-8

# In[6]:


# create a graph with two cycles glued along a path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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


plt.imshow(R, cmap='viridis', interpolation='nearest')
plt.title("Extended Effective Resistance for 2 Directed Cycles")
plt.colorbar()
plt.show()

plt.imshow(d1, cmap='viridis', interpolation='nearest')
plt.title("Hitting Probabilities B=1 for 2 Directed Cycles")
plt.colorbar()
plt.show()

plt.imshow(d12, cmap='viridis', interpolation='nearest')
plt.title("Hitting Probabilities B=1/2 for 2 Directed Cycles")
plt.colorbar()
plt.show()

pgraph = nx.from_numpy_array(P, parallel_edges=False)
nx.draw_networkx(pgraph, nx.spectral_layout(pgraph), node_size=30, node_color = "blue", with_labels = False, width = 1)
plt.title("2 Directed Cycles Graph")

