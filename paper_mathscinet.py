#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 13:54:32 2020

@author: minx
"""


import igraph
from tree_tools import *
from gibbsSampling import *
from grafting import *
import pickle


foo = Graph.Read_Ncol("data/ca-MathSciNet.mtx")
foo.to_undirected()
foo.simplify()

n0 = len(foo.vs)
m0 = len(foo.es)
graf = foo.clusters().giant()
n = len(graf.vs)
m = len(graf.es)


degs = graf.degree()
degs_sorted = -np.sort( - np.array(degs))

#res = gibbsTreeToConv(graf, K=1)
with open("pickles/mathscinet.pkl", "rb") as f:
    res, graf = pickle.load(f)

freq = res[0]
freq_sorted = -np.sort(-freq)
freq_args = np.argsort(-freq)

""" confidence set construction """
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


""" central subgraph """

#n_sm = max(sizes[2], 200)
n_sm = 200

graf.vs["size"] = 3
graf.vs["color"] = "darkgray"
graf.es["color"] = "rbga(1,1,1,0.4)"


graf.vs["label"] = ""
graf.vs["label_size"] = 16
graf.vs["label_dist"] = 2

for ii in range(sizes[0]):
    graf.vs[freq_args[ii]]["color"] = "orange"
    graf.vs[freq_args[ii]]["size"] = "10"
    graf.vs[freq_args[ii]]["label"] = round(freq_sorted[ii], 3)
    
    
graf_sub = graf.subgraph(freq_args[0:n_sm])

out_fig = "figs/mathscinet_subnet.eps"
my_layout = graf_sub.layout_fruchterman_reingold(niter=500)
visual_style = {}
visual_style["layout"] = my_layout
visual_style["edge_color"] = "gray"

plot(graf_sub, out_fig, **visual_style)


#with open("pickles/mathscinet.pkl", "wb") as f:
#    pickle.dump([res, graf], f)