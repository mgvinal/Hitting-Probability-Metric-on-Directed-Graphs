#!/usr/bin/env python
# coding: utf-8

# In[8]:


def HittingTimes_L3(M):
    # this program is based on the paper "Sharp Entrywise Perturbation Bounds for Markov Chains"
    # input: M is a stochastic probability transition matrix M
    # ouput: Q is a matrix of probablities of leaving i and then hitting j before returning back to i. Q will be used in 
    # conjunction with the invariant measure to find the normalized hitting probabilities matrix (Aht) in the next function
    
    # M must be a stochastic matrix. If the desired input A is nonstochastic, row normalization should be used to make the input
    # matrix stochastic. Alternatively, a similarity transformation involving the dominant right eigenvector of A could be used. 
    import numpy as np
    from numpy import linalg
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

