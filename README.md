# Hitting-Probability-Metric-on-Directed-Graphs
This code is the Python adaptation of the MATLAB code found in the paper ["A Metric on Directed Graphs and Markov Chains Based on Hitting Probabilities,"](https://marzuola.web.unc.edu/wp-content/uploads/sites/16865/2020/06/A_metric_on_the_state_space_of_Markov_chains_based_on_hitting_times.pdf) by Boyd, Fraiman, Marzuola, Mucha, Osting, and Weare. This paper addresses the fact that the shortest path and generalized effective resistance graph metrics have been applied to directed graphs in the same way as undirected graphs without any consideration for the obvious differences between the two types. It comes up with a new metric, more suited to analyzed directed graphs. The Hitting Probability Metric can be used on any strongly connected, directed graph. Possible applications for this metric include structure detection, dimension reduction, visualization, dynamics exploration, and multiscale cluster detection. Other applications involve nearest-neighbor search, new notions of graph curvature, Cheeger inequalities, and provable optimality of weak recovery for dense, directed communities. 

<img src="2_cycles.png" width="400"> <img src="d12_hm.png" width="400"> <img src="d1_hm.png" width="400"> <img src="exteffres2cyc.png" width="400">

<img src="tree.png" width="400"> <img src="d12t_hm.png" width="400"> <img src="d1t_hm.png" width="400"> <img src="exteffrestree.png" width="400">

## Description of Files: 
### Hitting Probability Metric on Directed Graphs includes 3 distinct functions with two examples.
1. [HittingTimes_L3.py](https://github.com/mgvinal/Hitting-Probability-Metric-on-Directed-Graphs/blob/main/HittingTimes_L3.py): returns a matrix of probabilities, Q, of leaving one node (i) and then hitting another node (j) before returning to the original node (i). It requires an input of a stochastic probability transition matrix M. The algorithm behind this function is from the paper ["Sharp Entrywise Perturbation Bounds for Markov Chains,"](https://arxiv.org/pdf/1410.1431.pdf) by Thiede, Koten, and Weare. The matrix of probabilities will be used in conjunction with the invariant measure to find the normalized hitting probabilities matrix in the next function. 
2. [get_Ahp.py](https://github.com/mgvinal/Hitting-Probability-Metric-on-Directed-Graphs/blob/main/get_Ahp.py): returns the normalized hitting probabilities matrix of the matrix of probabilities found in HittingTimes_L3(M). There are two inputs for this function: P and beta. P must be a stochastic probability transition matrix, as it is inputted into Hitting_TimesL3(M) within this function. If the desired input is a nonstochastic adjacency matrix, row normalization can be used to make the input array stochastic. Alternatively, a similarity transformation using the dominant right eigenvector of the desired input can be used. Beta is a scale parameter that must be between 0.5 and 1 (inclusive) to satisfy the triangle inequality and positivity requirements. Above 1, although these requirements are still met, the tightness of the triangle inequality decreases so much that the metric structure becomes less interesting, providing diminishing returns. Beta=0.5 is a pseudometric because the positivity requirement isn’t being met, but the triangle inequality is at its tightest point, providing fascinating insights into the metric structure of the graph in question. 
3. [extEffRes.py](https://github.com/mgvinal/Hitting-Probability-Metric-on-Directed-Graphs/blob/main/extEffRes.py): finds the generalized effective resistance from a probability transition matrix (A) for a directed or undirected graph. The output of this function (R) is not the metric in question; R^0.5 is. The algorithm behind this function is from the paper [A new notion of effective resistance for directed graphs---Part I: Definition and Properties"](https://doi.org/10.48550/arXiv.1310.5163) by F. Young, L. Scardovi, and N. E. Leonard. This function returns a typical metric for undirected graphs that has been applied to directed graphs in the past. It will serve as a helpful comparison to this new metric in the following examples. 
4. [ex_undirected_tree_script.py](https://github.com/mgvinal/Hitting-Probability-Metric-on-Directed-Graphs/blob/main/ex_undirected_tree_script.py): contains an example of a standard undirected tree graph. The extended effective resistance and hitting probabilities when beta=1 and 0.5 matrices are all computed and visualized via heatmaps. A graph visualization of this tree is also generated. This undirected graph example proves the relationship discovered in the paper in question: in an undirected graph, this new metric is directly proportional to the effective resistance when beta=1.  
5. [ex_two_directed_cycles_glued_together_script.py](https://github.com/mgvinal/Hitting-Probability-Metric-on-Directed-Graphs/blob/main/ex_two_directed_cycles_glued_together_script.py): contains the example used in the original paper. This graph is directed and consists of two directed 60-node-length cycles, connected along 5 common nodes. This example also undergoes calculations of its extended effective resistance and hitting probabilities when beta=1 and 0.5 matrices. These matrices are visualized using a heatmap, and a visualization of the graph is provided.  
## Funding Acknowledgment:
This project was supported by the National Science Foundation's (NSF) Focused Research Groups (FRG) Division of Mathematical Sciences (DMS) [grant number 2152289]. 
