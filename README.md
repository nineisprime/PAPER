PAPER
====

Implementation of Gibbs sampler for computing posterior root
probabilities under the PAPER (preferential attachment plus
Erdos--Renyi) model for random networks.

See details in the arXiv paper: https://arxiv.org/abs/2107.00153

[Documentation]

[Documentation]: https://nineisprime.github.io/PAPER/

Installation
-------------

	$ pip install PAPER


Usage
------

	>>> from PAPER.gibbsSampling import gibbsToConv
	>>> from PAPER.tree_tools import createNoisyGraph
	>>> graf = createNoisyGraph(n=100, m=200, alpha=0, beta=1, K=1)[0]
	>>> mcmc_out = gibbsSampling.gibbsToConv(graf, DP=False, method="full",
                       K=1, tol=0.1)
					   
See example.py for interpreting the inference output. Some sample
network datasets are provided.

Notes
------
* No preprocessing required on input graph. If the
  input graph is disconnected, the largest connected component is
  used.
* The algorithm
  performs roughly 1 outer Gibbs iteration in 1 second on a graph with
  10,000 edges. The number of iterations to convergence depends on the
  input graph.
