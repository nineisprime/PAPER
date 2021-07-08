#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 22:47:01 2021

@author: minx
"""


from tree_tools import *
import numpy as np
import pickle
from gibbsSampling import *
from grafting import *


ntrial = 200

alpha_ls = [0, 1, 3, 6, 1]
beta_ls = [1, 1, 1, 1, 0]

K = 1

n = 3000
m = 15000

res = np.zeros(shape=(ntrial, len(alpha_ls)))

for it in range(ntrial):
    print(it)
    for j in range(len(alpha_ls)):

        alpha = alpha_ls[j]
        beta = beta_ls[j]        

        graf, memb = createNoisyGraph(n, m, alpha=alpha, beta=beta, K=K)
        alphahat = estimateAlphaEM(graf, display=False)
        res[it, j] = alphahat
        
allmeans = np.apply_along_axis(np.mean, 0, res)
allsds = np.apply_along_axis(np.std, 0, res)