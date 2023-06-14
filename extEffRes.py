#!/usr/bin/env python
# coding: utf-8

# In[10]:


def extEffRes(A):
    # input: A is the probability transition matrix for a (possibly) directed graph
    # output: R is the square of the generalized effective resistance matrix as defined in "A New Notion of Effective 
    # Resistance for Directed Graphs-Part I: Definition and Properties". It is a symmetric matrix. 
    # all equations referenced in thie function are from the paper above
    # R itself is not a metric, we will want to use R^0.5 as the generalized effective resistance matrix
    import numpy as np
    import scipy
    from scipy import linalg
    
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

