#!/usr/bin/env python
# coding: utf-8

# In[5]:


# create an undirected tree graph as another geeometric set example
import igraph as ig
import numpy as np
from igraph import Graph
from sklearn.preprocessing import normalize
tgraph = Graph.Tree(115,5)
tmat = np.array(tgraph.get_adjacency().data)
#normalize the data so that the rows each sum to 1; need to do this to 
# correctly compute the Hitting Probability matrices and extended effective
# resistance
tmat = normalize(tmat, axis=1, norm='l1')

# compute extended effective resistance on the tree graph
R_t = extEffRes(tmat)   
R_t = np.sqrt(R_t)

# compute the hitting probability matrix when beta=1/2
dt_12 = -np.log(get_Ahp(tmat))
for i in range(0, 60):
    dt_12[i, i] = 0
    
# compute the hitting probability matrix when beta=1
dt_1 = -np.log(get_Ahp(tmat, 1))
for i in range(0, 60):
    dt_1[i, i] = 0
assert (np.nanmax(((dt_1 - np.transpose(dt_1)) / dt_1)) < 0.001)

