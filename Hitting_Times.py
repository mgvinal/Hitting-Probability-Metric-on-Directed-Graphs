#!/usr/bin/env python
# coding: utf-8

# In[1]:


def HittingTimes_L3(M):
    import numpy as np
    from numpy import linalg
    # Requires a stochastic matrix and outputs a matrix of probabilities of 
    # leaving i and then hitting j before returning back to i 
    N = M.shape[0]
    e1 = np.zeros((1, N))
    e1[0,0] = 1
    a1inv = np.eye(N) - M # Not A1inv yet, but this saves on memory
    a1inv[0,] = e1
    a1inv = np.linalg.inv(a1inv)
    Q = np.zeros((N, N))
    Q[:, 0] = np.divide(a1inv[:, 0], np.diag(a1inv))
    M = M @ a1inv
    
    detcj = (1 + np.diag(M)) * (1 - M[0, 0]) + M[:, 0] * np.transpose(M[0,])
    cjinv = np.zeros((2, 2, N))
    cjinv[0, 0, :] = (1 - M[0, 0]) / detcj
    cjinv[0, 1, :] = M[:, 0] / detcj
    cjinv[1, 0, :] = -(np.transpose(M[0,])) / detcj
    cjinv[1, 1, :] = (1 + np.diag(M)) / detcj
    
    M1 = np.double(np.zeros((N, 2, N)))
    M1[:, 0, :] = a1inv
    a = np.array(-a1inv[:, 0])
    a = a.reshape(N, 1)
    M1[:, 1, :] = np.tile(a, (1, N)) # so each column is the same
    M2 = np.zeros((2, N, N))
    M2[0, :, :] = np.transpose(M)
    M2[1, :, :] = np.transpose(np.tile(M[0,], (N, 1)))
    ac = np.zeros((N, 1))
    ad = np.zeros((N, 1))
    for j in range(1, N):
        
        ac = a1inv[:, j] - M1[..., j] @ cjinv[..., j] @ M2[:, j, j]
        st = np.transpose(M1[..., j]) * (cjinv[..., j] @ M2[..., j])
        ad = np.diag(a1inv) - np.transpose(st.sum(axis = 0))
        Q[:, j] = ac / ad
    Q = Q - np.diag(np.diag(Q))
    return Q

