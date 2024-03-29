#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from numpy import linalg
import scipy
from scipy import linalg
import igraph as ig
from igraph import Graph
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import networkx as nx

def HittingTimes_L3(M):
    # this program is based on the paper "Sharp Entrywise Perturbation Bounds for Markov Chains"
    # input: M is a stochastic probability transition matrix M
    # ouput: Q is a matrix of probablities of leaving i and then hitting j before returning back to i. Q will be used in 
    # conjunction with the invariant measure to find the normalized hitting probabilities matrix (Aht) in the next function
    
    # M must be a stochastic matrix. If the desired input A is nonstochastic, row normalization should be used to make the input
    # matrix stochastic. Alternatively, a similarity transformation involving the dominant right eigenvector of A could be used. 
    
    # all referenced equations are from the aforementioned paper
    
    # EQN 25
    
    N = M.shape[0]
    e1 = np.zeros((1, N))
    e1[0,0] = 1
    a1inv = np.eye(N) - M # Not A1inv yet, but this saves on memory
    a1inv[0,] = e1
    a1inv = np.linalg.inv(a1inv)
    
    # EQN 26
    
    Q = np.zeros((N, N))
    Q[:, 0] = np.divide(a1inv[:, 0], np.diag(a1inv))
    M = M @ a1inv
    
    # EQN 27
    
    # cjinv = C(j)^-1 formula 
    detcj = (1 + np.diag(M)) * (1 - M[0, 0]) + M[:, 0] * np.transpose(M[0,])
    cjinv = np.zeros((2, 2, N))
    cjinv[0, 0, :] = (1 - M[0, 0]) / detcj
    cjinv[0, 1, :] = M[:, 0] / detcj
    cjinv[1, 0, :] = -(np.transpose(M[0,])) / detcj
    cjinv[1, 1, :] = (1 + np.diag(M)) / detcj
    
    # M1[...,j] = A(1)^-1(ej - e1)
    M1 = np.double(np.zeros((N, 2, N)))
    M1[:, 0, :] = a1inv
    a = np.array(-a1inv[:, 0])
    a = a.reshape(N, 1)
    M1[:, 1, :] = np.tile(a, (1, N)) # so each column is the same
    
    # M2[:,j,j] = ([e_j^T],  [e_1^T])(SA(1)^-1)
    M2 = np.zeros((2, N, N))
    M2[0, :, :] = np.transpose(M)
    M2[1, :, :] = np.transpose(np.tile(M[0,], (N, 1)))
    
    # EQN 28
    
    # ac is the numerator = A(1)^-1_i,j
    # ad is the denominator = A(1)^-1_i,i
    ac = np.zeros((N, 1))
    ad = np.zeros((N, 1))
    for j in range(1, N):
        ac = a1inv[:, j] - M1[..., j] @ cjinv[..., j] @ M2[:, j, j]
        # st is an intermediate to finding ad
        st = np.transpose(M1[..., j]) * (cjinv[..., j] @ M2[..., j])
        ad = np.diag(a1inv) - np.transpose(st.sum(axis = 0))
        Q[:, j] = ac / ad
        
    # final step to find Q
    Q = Q - np.diag(np.diag(Q))
    return Q


# In[2]:


def get_Ahp(P, beta=0.5):
    # inputs: P must be a stochastic matrix, and beta must be a number between 0.5 and 1 and acts as a scale parameter. 
    # output: Aht is the normalized hitting probabilities matrix. The hitting probability metric can be found from this 
    # matrix by taking -log(Aht)
    
    # when the desired input is a nonstochastic adjacency matrix A, the matrix must first be transformed into the stochastic
    # matrix P by preferably normalizing all of the rows so they add up to one. Alternatively, one could conduct a similarity
    # transformation involving the dominant right eigenvector of A. 
    
    # Find the invariant measure Aht given the matrix of probabilities P found in HittingTimes_L3
    # this function is based on equation 1.3 in "A Metric on Directed Graphs and Markov Chains Based on Hitting Probabilities"
    Q = HittingTimes_L3(P)
    # Find the invariant measure:
    # G=digraph(P);
    # v=centrality(G,'pagerank','FollowProbability',1,'Tolerance',1e-5,'MaxIterations',2000,'Importance',double(G.Edges.Weight));
    # v = pagerank(P',1,1e-2);
    n = P.shape[0]
    [w, v] = scipy.sparse.linalg.eigsh(np.double(np.transpose(P)), k=1, M=None, sigma=1+1e-6, which='LM', v0=np.ones(n), ncv=None, maxiter=100000, tol=1e-3, return_eigenvectors=True, Minv=None, OPinv=None, mode='normal')
    v = (np.absolute(v) / linalg.norm(v, 1))
    vt = np.transpose(v)
    
    #Constructs the symmetric adjacency matrix M
    if beta == 0.5:
        Aht = scipy.sparse.spdiags(vt ** 0.5, diags=0, m=n, n=n) @ Q @ scipy.sparse.spdiags(vt ** (-0.5), diags=0, m=n, n=n)
    elif beta == 1:
        v_arr = v.flatten(order="F")
        v_diag = np.diag(v_arr)
        Aht = v_diag @ Q
    elif beta > 0.5: 
        Aht = scipy.sparse.spdiags(vt ** beta, diags=0, m=n, n=n) @ Q @ scipy.sparse.spdiags(vt ** (beta-1), diags=0, m=n, n=n) 
    else: 
        raise Exception("Beta chosen cannot be below 0.5.")
        # Beta must be between 0.5 and 1 to satisfy the triangle inequality(>=0.5) and positivity(>0.5). 
        # Beta=0.5 is a pseudometric and there exists a quotient graph on which the distance function becomes a metric.
        # Any value between 0.5 and 1 (exclusive) is an actual metric. 
        # At beta>=1, no real metric exists. However, beta=1 is still a useful scale parameter to use for some situations. 
            
    Aht = 0.5 * (Aht + np.transpose(Aht))
    Aht = Aht - np.diag(np.diag(Aht))
    return Aht


# In[3]:


def extEffRes(A):
    # input: A is the probability transition matrix for a (possibly) directed graph
    # output: R is the square of the generalized effective resistance matrix as defined in "A New Notion of Effective 
    # Resistance for Directed Graphs-Part I: Definition and Properties". It is a symmetric matrix. 
    # all equations referenced in thie function are from the paper above
    # R itself is not a metric, we will want to use R^0.5 as the generalized effective resistance matrix
    
    n = A.shape[0]
    d_out = A.sum(axis=1) # out degrees
    # L is the laplacian matrix
    #np.diag(d_out) os the diagonal matrix of node out degrees
    L = np.diag(d_out) - A # satisfies  L*ones(n,1) = 0
    # Pi and Q satisfy Q*Pi = Q
    # Pi is the orthogonal projection matrix
    Pi = np.eye(n) - ((1 / (n)) * np.ones(n)) 
    D, V = scipy.linalg.eigh(Pi)
    Q = np.transpose(V[:, 1:])
    # eqn 4 solves for the reduced Laplacian (rL)
    rL = Q @ L @ np.transpose(Q)
    # eqn 8 solves the Lyapunov equation for Sig
    Sig = linalg.solve_continuous_lyapunov(rL, np.eye(n - 1))
    # eqn 10 finds X, the generalized inverse of the Laplacian matrix
    X = 2 * (np.transpose(Q) @ Sig @ Q)
    R = np.zeros((n, n))
    for ii in range(0, n):
        for jj in range(ii, n):
            # eqn 3 finds r, the effective resistance between nodes ii and jj
            r = X[ii, ii] + X[jj, jj] - 2 * X[ii, jj]
            R[ii, jj] = r
            R[jj, ii] = r
    return R 


# In[6]:


# create an undirected tree graph as another geeometric set example

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
for i in range(0, 115):
    dt_12[i, i] = 0
    
# compute the hitting probability matrix when beta=1
dt_1 = -np.log(get_Ahp(tmat, 1))
for i in range(0, 115):
    dt_1[i, i] = 0
assert (np.nanmax(((dt_1 - np.transpose(dt_1)) / dt_1)) < 0.001)


# In[7]:


# heatmaps for generalized effective resistance, hitting probability metric for beta=1 and 0.5, and visualization of tree graph. 

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

