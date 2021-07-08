#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:12:04 2021

@author: minx
"""



import igraph
from tree_tools import *
from gibbsSampling import *
from grafting import *

import pickle



Kdistr = {}


foo = igraph.read("data/karate.gml")

foo.to_undirected()
foo.simplify()

n = len(foo.vs)
m = len(foo.es)

graf = foo

mcmc_res = gibbsGraftToConv(graf, Burn=50, M=200, DP=True,
                                   alpha=0, beta=0, size_thresh=0, tol=0.1)

Kdistr[0] = np.concatenate([np.array(mcmc_res[1][2]),
                           np.array(mcmc_res[2][2])])




foo = igraph.read("data/polblogs.gml")
foo.to_undirected()
foo.simplify()

n0 = len(foo.vs)
m0 = len(foo.es)
bar = foo.clusters().giant()
n = len(bar.vs)
m = len(bar.es)
graf = bar

mcmc_res = gibbsGraftToConv(graf, Burn=50, M=50, DP=True,
                                   alpha=0, beta=0, size_thresh=0.01, tol=0.1)

Kdistr[1] = np.concatenate([np.array(mcmc_res[1][2]),
                        np.array(mcmc_res[2][2])])


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

axs[0].set_title("Karate")
axs[1].set_title("Blog")

axs[0].set_xlabel("K (num of roots)")
axs[1].set_xlabel("K (num of roots)")

fig.set_size_inches(8, 3)
plt.tight_layout()
plt.show()
fig.savefig("figs/randomKreal.pdf")                        