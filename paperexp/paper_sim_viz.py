#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:00:49 2021

@author: minx
"""

import pickle
import numpy as np


""" 
Generating the table for the first simulation

"""


with open("pickles/paper_exp1.pkl", "rb") as f:
    res, alpha_ls, beta_ls, eps_ls = pickle.load(f)
    

## res has the following structure
## res[it, j, k, 0/1]
## k = 0/1/2 -> eps level
## j = 0/1/2 -> alpha/beta setting

allcov = np.apply_along_axis(np.mean, 0, 
                    res[0:300, :, :, :])




"""
Plot edge/node ratio vs. size of the conf set for the 
uniform attachment setting

"""
with open("pickles/paper_exp2.pkl", "rb") as f:
    res, edge_ratio_ls, n_ls, alpha, beta, K = pickle.load(f)
        
edge_ratio_ls = edge_ratio_ls[0:4]
res = res[:, :, 0:4, :]
    
meanarr = np.apply_along_axis(np.mean, 0, res)
sdarr = np.apply_along_axis(np.std, 0, res)

fiveksize = meanarr[0, :, 1]
tenksize = meanarr[1, :, 1]

fiveksd = sdarr[0, :, 1]
tenksd = sdarr[1, :, 1]

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=15)

fig, ax = plt.subplots()
#ax.plot(edge_ratio_ls, fiveksize, label="n=5000")
ax.errorbar(edge_ratio_ls, fiveksize, yerr=fiveksd, fmt='-o',
            label="n=5,000")
#ax.plot(edge_ratio_ls, tenksize, label="n=10,000")
ax.errorbar(edge_ratio_ls, tenksize, yerr=tenksd, fmt='-o',
            label="n=10,000")
ax.set_xlabel("number of edges = c * n log(n) for values of c")
ax.set_ylabel("conf set size")
ax.legend(loc="upper left")
plt.title("uniform attachment")
plt.tight_layout()
plt.savefig("figs/edge_size_ua.pdf")





"""
Plot edge/node ratio vs. size of the conf set for the 
linear preferential attachment setting

"""
with open("pickles/paper_exp3.pkl", "rb") as f:
    res, edge_ratio_ls, n_ls, alpha, beta, K = pickle.load(f)
        
meanarr = np.apply_along_axis(np.mean, 0, res)
sdarr = np.apply_along_axis(np.std, 0, res)

fiveksize = meanarr[0, :, 1]
tenksize = meanarr[1, :, 1]

fiveksd = sdarr[0, :, 1]
tenksd = sdarr[1, :, 1]

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=15)

fig, ax = plt.subplots()
#ax.plot(edge_ratio_ls, fiveksize, label="n=5000")
ax.errorbar(edge_ratio_ls, fiveksize, yerr=fiveksd, fmt='-o',
            label="n=5,000")
#ax.plot(edge_ratio_ls, tenksize, label="n=10,000")
ax.errorbar(edge_ratio_ls, tenksize, yerr=tenksd, fmt='-o',
            label="n=10,000")
ax.set_xlabel("number of edges = c * n sqrt(n) for values of c")
ax.set_ylabel("conf set size")
ax.legend()
plt.title("linear preferential attachment")
plt.tight_layout()
plt.savefig("figs/edge_size_lpa.pdf")


""" 
Generating the table for the 4th simulation

"""

with open("pickles/paper_exp4.pkl", "rb") as f:
    res, alpha_ls, beta_ls, eps_ls = pickle.load(f)
    

## res has the following structure
## res[it, j, k, 0/1]
## k = 0/1/2 -> eps level
## j = 0/1/2 -> alpha/beta setting

allcov = np.apply_along_axis(np.mean, 0, 
                    res[0:80, :, :, :])





""" Generating plot for 5th simulation
"""



import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib

nb = 16

matplotlib.rc('font', size=13)

fig, axs = plt.subplots(1, 2, sharey=False)

axs[0].set_xticks(ticks=[2, 5, 10, 20])
axs[1].set_xticks(ticks=[2, 5, 10, 20])
axs[0].hist(Kdistr[1], bins=nb, ec='black', density=True)
axs[1].hist(Kdistr[0], bins=nb, ec='black', density=True)

axs[0].set_title("UA")
axs[1].set_title("LPA")

axs[0].set_xlabel("K (num of roots)")
axs[1].set_xlabel("K (num of roots)")

fig.set_size_inches(8, 3)
plt.tight_layout()
plt.show()
fig.savefig("figs/randomKsim.pdf")


