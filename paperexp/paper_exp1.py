#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:54:58 2021

@author: minx
"""



from PAPER.tree_tools import *
import numpy as np
import pickle
from PAPER.gibbsSampling import *
from PAPER.grafting import *

n = 3000
K = 1
m = 7500

ntrials = 300

INIT = True

alpha_ls = [0, 1, 8]
beta_ls = [1, 0, 1]

eps_ls = [.01, .05, .2]


if (INIT):
    with open("pickles/paper_exp1.pkl", "rb") as f:
        res, alpha_ls, beta_ls, eps_ls = pickle.load(f)
else:
    res = np.zeros(shape=(ntrials, len(alpha_ls), len(eps_ls), 2))


for it in range(ntrials):
    
    for j in range(len(alpha_ls)):
        
        alpha = alpha_ls[j]
        beta = beta_ls[j]
        
        if (j > 0):
            continue
        else:
            res[it, j, :, 1] = 0
        #if (res[it, j, 0, 1] != 0):
        #    continue
        
        graf = createNoisyGraph(n, m, alpha=alpha, beta=beta, K=K)[0]
        
        mcmc_res = gibbsToConv(graf, Burn=20, M=60, DP=False, method="full",
                                   K=1, tol=0.1)
        
        freq = mcmc_res[0]
        sort_ix = np.argsort(-freq)
        
        sofar = 0
        conf_set = []
        
        #print(sort_ix[0:10])
        
        for ii in range(len(sort_ix)):
            
            for k in range(len(eps_ls)):
                
                if (res[it, j, k, 1] > 0):
                    continue
                
                if (sofar >= 1 - eps_ls[k]):
                    
                    res[it, j, k, 0] = (0 in sort_ix[0:ii])
                    res[it, j, k, 1] = ii
                
            sofar = sofar + freq[sort_ix[ii]]
        
        print("iter {0}  j {1}  cov1 {2}  cov2 {3}  cov3 {4}".format(it, j, 
                                                                     res[it, j, 0, 0],
                                                                     res[it, j, 1, 0],
                                                                     res[it, j, 2, 0]))
        
        with open("pickles/paper_exp1.pkl", 'wb') as f:
            pickle.dump([res, alpha_ls, beta_ls, eps_ls], f)
            
            
