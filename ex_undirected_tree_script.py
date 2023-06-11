#!/usr/bin/env python
# coding: utf-8

# In[2]:


# create an undirected tree graph as another geeometric set example
import igraph as ig
from igraph import Graph
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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

plt.imshow(R_t, cmap='viridis', interpolation='nearest')
plt.title("Extended Effective Resistance for a Tree")
plt.colorbar()
plt.show()

plt.imshow(dt_1, cmap='viridis', interpolation='nearest')
plt.title("Hitting Probabilities B=1 for a Tree")
plt.colorbar()
plt.show()

plt.imshow(dt_12, cmap='viridis', interpolation='nearest')
plt.title("Hitting Probabilities B=1/2 for a Tree")
plt.colorbar()
plt.show()

t_graph = tgraph.to_networkx()
plt.figure()
plt.title("Tree Graph")
nx.draw_networkx(t_graph, nx.kamada_kawai_layout(t_graph), node_size=30, node_color = "blue", with_labels = False, width = 1)
plt.show()

