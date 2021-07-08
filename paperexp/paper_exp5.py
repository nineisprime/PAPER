#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:54:58 2021

@author: minx
"""



from tree_tools import *
import numpy as np
import pickle
from gibbsSampling import *
from grafting import *


n = 1000

m = 1500

ntrials = 20

INIT = False

K_ls = [2, 2]

alpha_ls = [0, 1]
beta_ls = [1, 0]

eps_ls = [.01, .05, .2]


if (INIT):
    with open("pickles/paper_exp5.pkl", "rb") as f:
        res, alpha_ls, beta_ls, eps_ls = pickle.load(f)
else:
    res = np.zeros(shape=(ntrials, len(alpha_ls), len(eps_ls), 2))
    Kdistr = {}
    for j in range(len(alpha_ls)):
        Kdistr[j] = []


for it in range(ntrials):
    
    for j in range(len(alpha_ls)):
        
        alpha = alpha_ls[j]
        beta = beta_ls[j]
        
        if (res[it, j, 0, 1] != 0):
            continue
        
        K = K_ls[j]
        
        tmp = createNoisyGraph(n, m, alpha=alpha, beta=beta, K=K)
        graf = tmp[0]
        
        print(np.bincount(tmp[1]))
        
        mcmc_res = gibbsGraftToConv(graf, Burn=50, M=50, DP=True,
                                   alpha=0, beta=0, size_thresh=0, tol=0.1)
        
        Kdistr[j] = np.concatenate( [np.array(Kdistr[j]),
                                     np.array(mcmc_res[1][2]),
                                     np.array(mcmc_res[2][2])])
        
        allK = np.concatenate([np.array(mcmc_res[1][2]), 
                              np.array(mcmc_res[2][2]) ] )
        
        freq = mcmc_res[0]
        sort_ix = np.argsort(-freq)
        
        sofar = 0
        conf_set = []
        
        for ii in range(len(sort_ix)):
            
            for k in range(len(eps_ls)):
                
                if (res[it, j, k, 1] > 0):
                    continue
                
                if (sofar >= 1 - eps_ls[k]/mean(allK) ):
                    res[it, j, k, 0] = all([u in sort_ix[0:ii] for u in range(K)])
                    res[it, j, k, 1] = ii
                    
            sofar = sofar + freq[sort_ix[ii]]
        
        for k in range(len(eps_ls)):
            if (res[it, j, k, 1] == 0):
                res[it, j, k, 1] = n
                res[it, j, k, 0] = 1
        
        
        print("iter {0}  j {1}  cov1 {2}  cov2 {3}  cov3 {4}".format(it, j, 
                                                                     res[it, j, 0, 0],
                                                                     res[it, j, 1, 0],
                                                                     res[it, j, 2, 0]))
        
        with open("pickles/paper_exp5.pkl", 'wb') as f:
            pickle.dump([res, alpha_ls, beta_ls, eps_ls], f)
            
            


