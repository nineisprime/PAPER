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
import pandas as pd


import pickle

import wordcloud
import matplotlib.pyplot as plt

cities = pd.read_table("data/airports/global-cities.dat", sep="|", header=None)

name_map = {}

for i in range(cities.shape[0]):
    name_map[cities.loc[i][1]] = cities.loc[i][2]

foo = igraph.read("data/airports/global-net.dat")
foo.to_undirected()
foo.simplify()

foo.vs["myid"] = range(len(foo.vs))

n0 = len(foo.vs)
m0 = len(foo.es)
bar = foo.clusters().giant()
n = len(bar.vs)
m = len(bar.es)
graf = bar


with open("pickles/airport_random_K.pkl", "rb") as f:
    res, graf = pickle.load(f)
    
#res = gibbsGraftDP(graf, Burn=20, M=6000)

rev_id = {}
for i in range(n):
    rev_id[graf.vs[i]["myid"]] = i


tree_count = [0] * len(res[1])
for k in range(len(res[1])):
    tree_count[k] = sum(res[1][k])

tree_ord = np.argsort( -np.array( tree_count))

name_freq = {}
for k in tree_ord:
    
    if (sum(res[1][k]) < 60):
        continue
    
    name_freq[k] = {}
    print("")
    print(k)
    print("supp {0} ct {1}".format(sum(res[1][k] > 0), sum(res[1][k])))
    tmp1 = np.argsort(-res[1][k])
    
    tmp2 = res[1][k]/sum(res[1][k])
    for u in tmp1[0:40]:
        myname = name_map[graf.vs[int(u)]["myid"]]
        #print("{0} {1}".format(myname, round(tmp2[u],3)))
        name_freq[k][myname] = tmp2[u]


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % np.random.randint(0, 40)

for j in range(12):

    wc = wordcloud.WordCloud(background_color="white")
    wc.generate_from_frequencies(name_freq[tree_ord[j]])

    wc.recolor(color_func=grey_color_func, random_state=3)

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("figs/wordcloud" + str(j) + ".pdf")
    plt.show()



#with open("pickles/airport_random_K.pkl", "wb") as f:
#    pickle.dump([res, graf], f)
