#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 17:27:46 2021

@author: minx
"""

import PAPER.gibbsSampling as gibbsSampling
import PAPER.tree_tools as tree_tools
from igraph import *
import numpy as np

## Generate a PAPER graph
graf = tree_tools.createNoisyGraph(n=100, m=200, alpha=0, beta=1, K=1)[0]

## Run Gibbs sampler
mcmc_out = gibbsSampling.gibbsToConv(graf, DP=False, method="full",
                       K=1, alpha=0, beta=1, tol=0.05)


## Show nodes with top 10 posterior root prob:
rootprob = mcmc_out[0]
rootprob_args = np.argsort(-rootprob)
rootprob_sorted = - np.sort(-rootprob)

print( rootprob_args[0:10] )
print( [round(x, 2) for x in rootprob_sorted[0:10] ] )
