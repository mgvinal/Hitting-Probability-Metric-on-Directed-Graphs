#!/usr/bin/env python
# coding: utf-8

# In[3]:


def extEffRes(A):
    import numpy as np
    import scipy
    from scipy import linalg
    # the input A is the probability transition matrix for a (possibly)
    # directed graph
    # this function outputs the extended effective resistance as defined
    # in Young, Scardovi, and Leonard (2013). The notation below is from
    # this paper
    # R itself is not a metric, we will want to use R^0.5
    n = A.shape[0]
    d_out = A.sum(axis=1) # out degrees
    L = np.diag(d_out) - A # satisfies  L*ones(n,1) = 0
    # Pi and Q satisfy Q*Pi = Q
    Pi = np.eye(n) - ((1 / (n)) * np.ones(n)) # projection matrix
    D, V = scipy.linalg.eigh(Pi)
    Q = np.transpose(V[:, 1:])
    # reduced Laplacian
    rL = Q @ L @ np.transpose(Q)
    # solve the Lyapunov equation for Sig
    Sig = linalg.solve_continuous_lyapunov(rL, np.eye(n - 1))
    # X plays the role of L^{-1}
    X = 2 * (np.transpose(Q) @ Sig @ Q)
    R = np.zeros((n, n))
    for ii in range(0, n):
        for jj in range(ii, n):
            r = X[ii, ii] + X[jj, jj] - 2 * X[ii, jj]
            R[ii, jj] = r
            R[jj, ii] = r
    return R 

