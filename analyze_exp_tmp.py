#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:59:43 2021

@author: minx
"""

import pickle
import numpy as np


with open("pickles/paper_exp2c.pkl", "rb") as f:
    res = pickle.load(f)
    
res = res[0]
np.apply_along_axis(np.mean, 0, res)


## res has the following structure
## res[it, j, k, 0/1]
## k = 0/1/2 -> eps level
## j = 0/1/2 -> alpha/beta setting


#foo = res[0:200, 0, 0, :]

#file_ls = ["a"]
file_ls = ["a", "b", "c", "d", "e", "f"]
res = np.zeros(shape=(300, 2, 3, 2))
#file_ls = ["a", "b", "c", "d", "e"]
#res = np.zeros(shape=(300, 2, 3, 2))

ix = 0
for i in range(len(file_ls)):
    
    with open("pickles/paper_exp4" + file_ls[i] + ".pkl", "rb") as f:
        res2, a, b, e = pickle.load(f)

    for j in range(300):
        if (res2[j, 1, 0, 1] > 0):
            res[ix, :, :, :] = res2[j, :, :, :]
            ix = ix+1
        else:
            break
print(ix)
np.apply_along_axis(np.mean, 0, 
                    res[0:ix, :, :, :])





with open("pickles/paper_exp2.pkl", "rb") as f:
    res, a, b, c, d, e = pickle.load(f)
    
np.apply_along_axis(np.mean, 0, res)




with open("pickles/paper_exp3.pkl", "rb") as f:
    res, a, b, c, d, e = pickle.load(f)
    
    
np.apply_along_axis(np.mean, 0, res)