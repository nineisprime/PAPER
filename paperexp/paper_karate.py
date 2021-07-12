#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:54:32 2020

@author: minx


To illustrate tree edges:
    
    1. uncomment code setting tree edges to red
    2. use graftree for layout
    3. change file name



"""

from PAPER.tree_tools import *
import igraph
import numpy as np
import pickle
from PAPER.gibbsSampling import *
from PAPER.grafting import *

graf = igraph.read("data/karate.gml")

n = len(graf.vs)
m = len(graf.es)

graf.to_undirected()
graf.simplify()


## true community 1
comm1 = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21]
graf.vs["value"] = 1
for u in comm1:
    graf.vs[u]["value"] = 0

""" analysis for K=1 roots """
K=1

res = gibbsToConv(graf, DP=False, K=K, alpha=0, method="full",
                      beta=0, Burn=60, M=100, tol=0.005)

freq = res[0]
freq_sorted = -np.sort(-freq)
sorted_ixs = np.argsort(-freq)



""" analysis for K=2 roots """
K=2

allres = gibbsToConv(graf, DP=False, K=K, alpha=0, method="full",
                      beta=0, Burn=60, M=100, tol=0.001)

freq = allres[0]
freq_sorted = -np.sort(-freq)
freq_args = np.argsort(-freq)

res = allres[1]

""" confidence set constructions """


eps_ls = [0.4, 0.2, 0.05, 0.01]
sizes = [0] * len(eps_ls)

sofar = 0
for ii in range(n):
    for j in range(len(eps_ls)):
        
        if (sizes[j] > 0):
            continue
        
        if (sofar >= 1 - eps_ls[j]/K):
            sizes[j] = ii
            
    sofar = sofar + freq_sorted[ii]
        
print("Confidence set levels {0}".format(eps_ls))
print("Confidence set sizes {0}".format(sizes))

""" misclustering error """

coo = res[2]
coo = coo/(coo[0,0] + coo[0, 1])

comm = coo[:, 0] - coo[:, 1]

for i in range(n):
    x = (comm[i] + 1)/2 ## val between 0 or 1
    #print(x)
    if (x > 0.5):
        mycolor = (1, 1.8*(1-x) + 0.1, 1.8*(1-x) + 0.1)
    else:
        mycolor = (x*1.8 + 0.1, x*1.8 + 0.1, 1)
    #print(mycolor)
    graf.vs[i]["color"] = mycolor


my_assign = np.array(comm > 0)
gold_assign = np.array(graf.vs["value"])
tot_err = sum(np.abs( my_assign - gold_assign ))
tot_err = min(tot_err, n-tot_err)

print("Misclustering error: {0}".format(tot_err/n))



""" full plot """


graf.es["color"] = "rgba(1,1,1,0.3)"
graf.es["width"] = 3

graf.vs["size"] = 20
graf.vs["label"] = ""

nconf = sizes[0]
print(sum(freq_sorted[0:nconf]))
print(freq_args[0:nconf])

for i in range(nconf):
    u = int(freq_args[i])
    if (u == 0 or u == 33):
        graf.vs[u]["shape"] = "rectangle"
    graf.vs[u]["label"] = round(freq_sorted[i], 3)*K
    graf.vs[u]["label_dist"] = 2
    graf.vs[u]["label_size"] = 23
    


out_fig = "figs/karate_net1.eps"
cur_layout = graf.layout_fruchterman_reingold(niter=500)
visual_style = {}
visual_style["layout"] = cur_layout

plot(graf, out_fig, **visual_style)



""" plotting the trees"""


graf.es["color"] = "rgba(1,1,1,0.3)"
graf.es["width"] = 2

graf.vs["size"] = 20

graftree = graf.copy()
graftree.delete_edges([mye for mye in graftree.es if not mye["tree"]])

for mye in graf.es:
    if mye["tree"]:
        mye["color"] = "red"

out_fig = "figs/karate_latent_tree1.eps"
tree_layout1 = graftree.layout_fruchterman_reingold(niter=500)

plot(graf, out_fig, layout=tree_layout1)




graf.es["color"] = "rgba(1,1,1,0.3)"

tmp = gibbsTreeToConv(graf, DP=False, K=K, alpha=0, 
                      beta=0, Burn=60, M=100, tol=0.01)

graftree = graf.copy()
graftree.delete_edges([mye for mye in graftree.es if not mye["tree"]])

for mye in graf.es:
    if mye["tree"]:
        mye["color"] = "red"

out_fig = "figs/karate_latent_tree2.eps"
tree_layout2 = graftree.layout_fruchterman_reingold(niter=500)

plot(graf, out_fig, layout=tree_layout2)



""" save layouts """
#with open("pickles/karate.pkl", "wb") as f:
#    pickle.dump([res, cur_layout, tree_layout1, tree_layout2], f)
