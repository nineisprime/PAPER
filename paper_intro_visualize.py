#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:57:29 2020

@author: minx
"""
from igraph import *

from tree_tools import *
import numpy as np
import pickle
from gibbsSampling import *
from grafting import *

# Generate noisy network
# save layout according to the true tree


n = 80
m = 100

baz = createPATree(n, alpha=2, beta=1)[0]

my_layout = baz.layout_fruchterman_reingold(niter=200)

""" Save layout """

visual_style = {}
visual_style["layout"] = my_layout
baz.vs["color"] = "Gray"

baz.vs["size"] = 8
baz.vs["label_dist"] = 0
baz.vs["label_size"] = 18
baz.vs["label_color"] = "Red"
baz.vs["label"] = ""
baz.vs[0:10]["size"] = 23
baz.vs[0:10]["color"] = "White"
baz.vs[0:10]["label"] = range(1, 11)


edge_ls = [(e.source, e.target) for e in baz.es]

addRandomEdges(baz, m)

baz.es["color"] = "rgba(1,1,1,0.3)"

for e in baz.es:
    if ((e.source, e.target) in edge_ls or (e.target, e.source) in edge_ls):
        e["color"] = "red"



baz.es["width"] = 2

"""Make first plot"""
out_fig = "figs/intro1.eps"

plot(baz, out_fig, **visual_style)



res = gibbsTreeToConv(baz, DP=False, K=1)

freq = res[0]

freq_ord = np.argsort(-freq)

conf_set1 = []
conf_set2 = []

cur_lvl = 0
ii = 0
while (cur_lvl < 0.95):
    conf_set1.append(freq_ord[ii])
    if (cur_lvl < 0.8):
        conf_set2.append(freq_ord[ii])
    
    cur_lvl = cur_lvl + freq[freq_ord[ii]]
    ii = ii + 1

"""Make second plot"""
out_fig = "figs/intro2.eps"
baz.vs["size"] = 6
baz.es["color"] = "Gray"
baz.vs["color"] = "Dark Gray"
baz.vs["label"] = ""
baz.vs[conf_set2]["size"] = 14
baz.vs[conf_set2]["color"] = "Green"
plot(baz, out_fig, **visual_style)


"""


