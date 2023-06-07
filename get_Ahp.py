#!/usr/bin/env python
# coding: utf-8

# In[2]:


def get_Ahp(P, beta=0.5):
    import numpy as np
    import scipy
    from scipy import linalg
    # Find the invariant measure given the matrix of probabilities found in
    # HittingTimes_L3
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
    # add an else statement for when beta doesn't equal 0.5 or 1 
    Aht = 0.5 * (Aht + np.transpose(Aht))
    Aht = Aht - np.diag(np.diag(Aht))
    return Aht

