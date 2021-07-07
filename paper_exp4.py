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

n = 700

m = 1000

ntrials = 300

INIT = True

K_ls = [2, 2]

alpha_ls = [0, 1]
beta_ls = [1, 0]

eps_ls = [.01, .05, .2]


if (INIT):
    with open("pickles/paper_exp4.pkl", "rb") as f:
        res, alpha_ls, beta_ls, eps_ls = pickle.load(f)
else:
    res = np.zeros(shape=(ntrials, len(alpha_ls), len(eps_ls), 2))


for it in range(ntrials):
    
    for j in range(len(alpha_ls)):
        
        alpha = alpha_ls[j]
        beta = beta_ls[j]
        
        if (res[it, j, 0, 1] != 0):
            continue
        
        K = K_ls[j]
        
        
        while (True):
            
            tmp = createNoisyGraph(n, m, alpha=alpha, beta=beta, K=K)
            graf = tmp[0]
            
            bar = graf.clusters().giant()
            n2 = len(bar.vs)
            if (n2 == n):
                break
        
        
        
        true_sizes = np.bincount(tmp[1])
        
        print(true_sizes)
        
        if (min(true_sizes) < 0.03 * n):
            mytol = 0.001
            MAXITER = 40
        else:
            mytol = 0.005
            MAXITER = 20
        
        mcmc_res = gibbsGraftToConv(graf, Burn=100, M=100, DP=False,
                                   K=K, alpha=0, beta=0, 
                                   size_thresh=0, tol=mytol, MAXITER=MAXITER)
        
        freq = mcmc_res[0]
        sort_ix = np.argsort(-freq)
        
        sofar = 0
        conf_set = []
        
        for ii in range(len(sort_ix)):
            
            for k in range(len(eps_ls)):
                
                if (res[it, j, k, 1] > 0):
                    continue
                
                if (sofar >= 1 - eps_ls[k]/K ):
                    res[it, j, k, 0] = all([u in sort_ix[0:ii] for u in range(K)])
                    res[it, j, k, 1] = ii
                
            sofar = sofar + freq[sort_ix[ii]]
        
        print("iter {0}  j {1}  cov1 {2}  cov2 {3}  cov3 {4}".format(it, j, 
                                                                     res[it, j, 0, 0],
                                                                     res[it, j, 1, 0],
                                                                     res[it, j, 2, 0]))
        
        with open("pickles/paper_exp4.pkl", 'wb') as f:
            pickle.dump([res, alpha_ls, beta_ls, eps_ls], f)
            
            