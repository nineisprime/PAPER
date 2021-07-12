#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:54:32 2020

@author: minx
"""

from PAPER.tree_tools import *
import igraph
import numpy as np
import pickle
from PAPER.gibbsSampling import *

graf = igraph.read("data/flu_net.gml")

n = len(graf.vs)
m = len(graf.es)

graf.to_undirected()
graf.simplify()

    

""" single root analysis for flu network """

res = gibbsToConv(graf, DP=False, K=1, alpha=0, 
                      beta=0, Burn=60, M=100, tol=0.01, method="full")

freq = res[0]
freq_sorted = -np.sort(-freq)
freq_args = np.argsort(-freq)




""" construct confidence sets """

eps_ls = [0.4, 0.2, 0.05, 0.01]
sizes = [0] * len(eps_ls)

sofar = 0
for ii in range(n):
    for j in range(len(eps_ls)):
        
        if (sizes[j] > 0):
            continue
        
        if (sofar >= 1 - eps_ls[j]):
            sizes[j] = ii
            
    sofar = sofar + freq_sorted[ii]
        
print("Confidence set levels {0}".format(eps_ls))
print("Confidence set sizes {0}".format(sizes))


""" correlation with betweenness """


btw = graf.betweenness(range(n), directed=False)

btw_args = np.argsort(-np.array(btw))
btw_sorted = np.sort(-np.array(btw))

import scipy.stats
tau, pval = scipy.stats.kendalltau(btw, freq)
print("Kendall tau: {0}".format(round(tau, 4)))


print("Top 10 nodes by betweeness: {0}".format(btw_args[0:10]))
print("Top 10 nodes by root prob: {0}".format(freq_args[0:10]))


""" Plot full graph """

graf.es["color"] = "rgba(1,1,1,0.3)"
graf.es["width"] = 2

graf.vs["size"] = 20

graf.vs["color"] = "gray"
graf.vs["label"] = ""
graf.vs["label_dist"] = 0
graf.vs["label_size"] = 0


for i in range(sizes[0]):
    u = int(freq_args[i])
    graf.vs[u]["color"] = "green"
    graf.vs[u]["label"] = round(freq_sorted[i], 3)
    graf.vs[u]["label_dist"] = 2
    graf.vs[u]["label_size"] = 23

graf.vs[0]["shape"] = "rectangle"

out_fig = "figs/flu_net1.eps"
cur_layout = graf.layout_fruchterman_reingold(niter=500)

plot(graf, out_fig, layout=cur_layout)


""" plot latent trees """
##

graf.es["color"] = "rgba(1,1,1,0.3)"

graftree = graf.copy()
graftree.delete_edges([mye for mye in graftree.es if not mye["tree"]])

for mye in graf.es:
    if mye["tree"]:
        mye["color"] = "red"

tree_layout1 = graftree.layout_fruchterman_reingold(niter=500)
out_fig = "figs/flu_latent_tree1.eps"
plot(graf, out_fig, layout=tree_layout1)



graf.es["color"] = "rgba(1,1,1,0.3)"
gibbsToConv(graf, DP=False, K=1, alpha=0, 
                      beta=0, Burn=60, M=100, tol=0.01, method="full")
graftree = graf.copy()
graftree.delete_edges([mye for mye in graftree.es if not mye["tree"]])

for mye in graf.es:
    if mye["tree"]:
        mye["color"] = "red"

tree_layout2 = graftree.layout_fruchterman_reingold(niter=500)
out_fig = "figs/flu_latent_tree2.eps"
plot(graf, out_fig, layout=tree_layout2)


""" pickle """

with open("pickles/flu_net.pkl", "wb") as f:
    pickle.dump([res, cur_layout, tree_layout1, tree_layout2], f)
