#!/usr/bin/env python
# coding: utf-8

# In[9]:


def get_Ahp(P, beta=0.5):
    # inputs: P must be a stochastic matrix, and beta must be a number between 0.5 and 1 and acts as a scale parameter. 
    # output: Aht is the normalized hitting probabilities matrix. The hitting probability metric can be found from this 
    # matrix by taking -log(Aht)
    
    # when the desired input is a nonstochastic adjacency matrix A, the matrix must first be transformed into the stochastic
    # matrix P by preferably normalizing all of the rows so they add up to one. Alternatively, one could conduct a similarity
    # transformation involving the dominant right eigenvector of A. 
    import numpy as np
    import scipy
    from scipy import linalg
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

