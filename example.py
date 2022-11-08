#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 17:27:46 2021

@author: minx
"""

from PAPER.gibbsSampling import gibbsToConv
from PAPER.tree_tools import createNoisyGraph
import numpy as np
import igraph

## Generate a PAPER graph
graf = createNoisyGraph(n=200, m=500, alpha=0, beta=1, K=2)[0]
# graf = igraph.read("data/flu_net.gml")  # alternatively, read graph from input file


## Run Gibbs sampler
mcmc_out = gibbsToConv(graf, DP=True, method="full",
                                      tol=0.01)



## Show nodes with top 10 posterior root prob:
rootprob = mcmc_out[0]
rootprob_args = np.argsort(-rootprob)
rootprob_sorted = - np.sort(-rootprob)

print( rootprob_args[0:10] )
print( [round(x, 2) for x in rootprob_sorted[0:10] ] )
