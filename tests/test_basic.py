#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:54:58 2021

@author: minx
"""


from PAPER.tree_tools import createNoisyGraph
import numpy as np
import pickle
from PAPER.gibbsSampling import gibbsToConv
from PAPER.estimateAlpha import estimateAlphaEM
import igraph

def test_gibbs_single_root():

    n = 50
    K = 1
    m = 53
    
    alpha = 0
    beta = 1

    graf = createNoisyGraph(n, m, alpha=alpha, beta=beta, K=K)[0]
        
    res = gibbsToConv(graf, Burn=20, M=40, DP=False, method="full",
                                   K=1, alpha=alpha, beta=beta, tol=0.05)
        
    assert abs(sum(res[0]) - 1) < 1e-3
            
            
    res2 = gibbsToConv(graf, Burn=20, M=40, DP=False, method="collapsed",
                                   K=1, alpha=alpha, beta=beta, tol=0.05)
    
    assert sum( np.abs( res[0] - res2[0] ))/2 < 0.2
    
    
    
def test_gibbs_single_root_with_estimation():
    
    n = 300
    K = 1
    m = 500
    
    alpha = 1
    beta = 1

    graf = createNoisyGraph(n, m, alpha=alpha, beta=beta, K=K)[0]
    alphahat = estimateAlphaEM(graf, display=False)
    assert abs(alphahat - alpha) < 3


    n = 50
    m = 52
    graf = createNoisyGraph(n, m, alpha=alpha, beta=beta, K=K)[0]
        
    res = gibbsToConv(graf, Burn=20, M=40, DP=False, method="full",
                                   K=1, alpha=0, beta=0, tol=0.02)
        
    assert abs(sum(res[0]) - 1) < 1e-3
            
            
    res2 = gibbsToConv(graf, Burn=20, M=40, DP=False, method="collapsed",
                                   K=1, alpha=0, beta=0, tol=0.02)
    
    assert sum( np.abs( res[0] - res2[0] ))/2 < 0.2    
    
    
    
def test_fixed_K_roots():
    
    n = 50
    K = 2
    m = 70
    
    alpha = 1
    beta = 1

    graf = createNoisyGraph(n, m, alpha=alpha, beta=beta, K=K)[0]
        
    res = gibbsToConv(graf, Burn=20, M=40, DP=False, method="full",
                                   K=K, alpha=alpha, beta=beta, tol=0.05,
                                   birth_thresh=1)
        
    assert abs(sum(res[0]) - 1) < 1e-3
            
            
    res2 = gibbsToConv(graf, Burn=20, M=40, DP=False, method="collapsed",
                                   K=K, alpha=alpha, beta=beta, tol=0.05,
                                   birth_thresh=1)
    
    assert sum( np.abs( res[0] - res2[0] ))/2 < 0.3
    
    
def test_random_K_roots():
    
    n = 50
    K = 2
    m = 70
    
    alpha = 1
    beta = 1

    graf = createNoisyGraph(n, m, alpha=alpha, beta=beta, K=K)[0]
        
    res = gibbsToConv(graf, Burn=20, M=40, DP=True, method="full",
                                   alpha=alpha, beta=beta, tol=0.05,
                                   birth_thresh=1)
        
    assert abs(sum(res[0]) - 1) < 1e-3
            
            
    res2 = gibbsToConv(graf, Burn=20, M=40, DP=True, method="collapsed",
                                   alpha=alpha, beta=beta, tol=0.05,
                                   birth_thresh=1)
    
    assert sum( np.abs( res[0] - res2[0] ))/2 < 0.3


def test_flu_net():

    graf = igraph.read("data/flu_net.gml")
    res = gibbsToConv(graf, Burn=20, M=40, DP=False, K=1, method="full",
                      tol=0.05)

    rootprobs = res[0]
    assert abs(rootprobs[0] - 0.11) < 0.4
