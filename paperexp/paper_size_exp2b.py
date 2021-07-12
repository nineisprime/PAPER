#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:51:04 2020

@author: minx
"""
## Uniform attachment
## 
## edge_ratio varies
##
## n stays constant
## 
## 
##
import sys
sys.path.append('../')
print(sys.path)


from PAPER.tree_tools import *
import numpy as np
import pickle
from PAPER.gibbsSampling import *
from PAPER.grafting import *

n_ls = [5000, 10000]
alpha = 1
beta = 0
K = 1

conf_lvl = 0.95

ntrials = 20
edge_ratio_ls = np.array([0.15, 0.2, 0.4, 0.6, 0.8, 1])
print(edge_ratio_ls)


""" 
edge_ratio_ls = [1.1, 1.5, 2, 3, 4]
"""

INIT = True

if (INIT): 
    with open("../pickles/paper_exp2b.pkl", "rb") as f:
        res, edge_ratio_ls, n_ls, alpha, beta, K = pickle.load(f)
else:
    res = np.zeros(shape=(ntrials, len(n_ls), len(edge_ratio_ls), 2))


    
for j in range(len(edge_ratio_ls)):
    for i in range(len(n_ls)):    
        for it in range(ntrials):

            if (j != 4 or i != 1):
                continue

            if (res[it, i, j, 1] != 0):
                print((it, i, j, res[it, i, j, 1]))
                continue
                
            n = n_ls[i]
            
            edge_ratio = np.log(n) * edge_ratio_ls[j]
        
            m = int(n * edge_ratio)
        
            graf = createNoisyGraph(n, m, alpha=alpha, beta=beta, K=K)[0]
        
            mcmc_res = gibbsToConv(graf, Burn=20, M=30, DP=False,
                                       K=1, alpha=0, beta=0, tol=0.1, MAXITER=60, method="full")
        
            freq = mcmc_res[0]
            sort_ix = np.argsort(-freq)
        
            sofar = 0
            conf_set = []
            for jj in sort_ix:
            
                if (sofar > conf_lvl):
                    break
            
                cur_prob = freq[jj]
                conf_set.append(jj)
                sofar = sofar + cur_prob
        
            print("iter {0}  ratio {1}  n {2}  size {3}  cov {4}".format(it, round(edge_ratio, 3), n, len(conf_set), (0 in conf_set)))
        
            res[it, i, j, 0] = (0 in conf_set)
            res[it, i, j, 1] = len(conf_set)

            with open("../pickles/paper_exp2b.pkl", 'wb') as f:
                pickle.dump([res, edge_ratio_ls, n_ls, alpha, beta, K], f)

#    with open("pickles/paper_exp1.pkl", 'wb') as f:
#        pickle.dump([res, edge_ratio_ls, n, a, b, K], f)

