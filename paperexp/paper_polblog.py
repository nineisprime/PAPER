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

foo = igraph.read("data/polblogs.gml")
foo.to_undirected()
foo.simplify()

n0 = len(foo.vs)
m0 = len(foo.es)
bar = foo.clusters().giant()
n = len(bar.vs)
m = len(bar.es)
graf = bar
K=2

allres = gibbsTreeToConv(graf, M=400, K=K, tol=0.001)

#with open("pickles/polblog.pkl", "rb") as f:
#    tmp, a, b, c, d = pickle.load(f)


res = allres[1]

freq = allres[0]
#freq = freq/sum(freq)
freq_sorted = -np.sort(-freq)
freq_args = np.argsort(-freq)

""" confidence set construction """

eps_ls = [0.4, 0.2, 0.05, 0.01, 0.001]
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


""" clustering error """
coo = res[2]
coo = coo/(coo[0,0] + coo[0, 1])

comm = -(coo[:, 0] - coo[:, 1])

for i in range(n):
    x = (comm[i] + 1)/2 ## val between 0 or 1
    #print(x)
    if (x > 0.5):
        mycolor = (1, 1.8*(1-x) + 0.1, 1.8*(1-x) + 0.1)
    else:
        mycolor = (x*1.8 + 0.1, x*1.8 + 0.1, 1)
    graf.vs[i]["color"] = mycolor

my_assign = np.array(comm > 0)
gold_assign = np.array(graf.vs["value"])
tot_err = sum(np.abs( my_assign - gold_assign ))
tot_err = min(tot_err, n-tot_err)

print("Misclustering error: {0}".format(tot_err/n))

topnodes = freq_args[0:400]
#topnodes = freq_args[0:sizes[2]]
tot_err2 = sum(np.abs(my_assign[topnodes] - gold_assign[topnodes]))
tot_err2 = min(tot_err2, len(topnodes)-tot_err2)
print("Misclustering error for top {0} nodes: {1}".format(len(topnodes), tot_err2/len(topnodes)))


""" Plotting whole graph """

graf.vs["size"]=5
graf.vs["label"] = ""
graf.es["color"] = "rgba(1,1,1,0.3)"
graf.es["width"] = 1

for ii in range(sizes[2]):
    graf.vs[freq_args[ii]]["size"] = "10"

cur_layout = graf.layout_fruchterman_reingold()

out_fig = "figs/polblog.eps"

visual_style = {}
visual_style["layout"] = cur_layout

plot(graf, out_fig, **visual_style)



""" Plotting an example tree """
graftree = graf.copy()
graftree.delete_edges([mye for mye in graftree.es if not mye["tree"]])

for mye in graf.es:
    if mye["tree"]:
        mye["color"] = "black"
    else:
        mye["color"] = "rgba(1,1,1,0.02)"
out_fig = "figs/polblog_tree.eps"        
tree_layout = graftree.layout_fruchterman_reingold(niter=300)
plot(graf, out_fig, layout=tree_layout)



#with open("pickles/polblog.pkl", "wb") as f:
#    pickle.dump([res, graf, cur_layout, tree_layout], f)