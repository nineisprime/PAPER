#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:10:19 2020

@author: minx
"""

import time
from random import *
from igraph import *
import numpy as np
import collections
import scipy.optimize
#from numpy.random import *

"""
README 
important invariants in the code
graf.vs has attribute "pa", "subtree_size"
graf.es has attribute "tree"

function "wilsonTree(graf)" adds "tree" attribute
function "countSubtreeSizes(graf, root)" require "tree" attribute 
      on the edges and creates "subtree_size" and "pa"
"""


def getAllTreeDeg(graf):
    n = len(graf.vs)
    degs = [0] * n
    for mye in graf.es:
        if (mye["tree"]):
            degs[mye.source] = degs[mye.source] + 1
            degs[mye.target ] = degs[mye.target] + 1
    degs = np.array(degs)
    return(degs)




def treeDegree(graf, v):
    
    edge_ixs = graf.incident(v)
    
    deg = sum([e["tree"] for e in graf.es[edge_ixs]])
    
    return(deg)

    
"""    
ancestors of utilde, avoiding u

IN: "u" a node to avoid, omit argument to always get the 
        ancestors of utilde               
OUT: return -1 if parents of utilde trace to u, 
          otherwise return list of parents of utilde, including utilde, including the root      

Note: root node is one whose "pa" attribute is None
"""
def getAncestors(graf, utilde, u = None):
    
    #assert graf.vs[utilde]["pa"] != None
    
    cur_node = utilde
    ants = [cur_node]
    
    while (True):
        my_pa = graf.vs[cur_node]["pa"]
        
        if (my_pa == None):
            return(ants)

        assert graf.vs[cur_node]["subtree_size"] < graf.vs[my_pa]["subtree_size"]
        
        if (my_pa == u):
            return(-1)
        else:
            ants.append(my_pa)
            cur_node = my_pa
        
"""


"""            
def otherNode(my_edge, node_ix):
    if (my_edge.source == node_ix):
        return(my_edge.target)
    else:
        return(my_edge.source)      


"""    
Creates a tree from "root" node by breadth-first-search

NOTE: modifies "graf", adds "tree" attribute to the edges

INPUT: "graf" igraph object
        "root" beginning of the BFS
OUTPUT: NONE 
"""
def bfsTree(graf, root=0):
    n = len(graf.vs)
    
    graf.vs["marked"] = False
    graf.es["tree"] = False
    
    graf.vs[root]["marked"] = True
    
    myqueue = [root]
    
    while (len(myqueue) > 0):
        v = myqueue.pop(0)
        
        edge_ixs = graf.incident(v)
        shuffle(edge_ixs)
        
        for eid in edge_ixs:
            my_e = graf.es[eid]
            u = otherNode(my_e, v)
            
            if (not graf.vs[u]["marked"]):
                my_e["tree"] = True
                graf.vs[u]["marked"] = True
                myqueue.append(u)
                
"""        
Creates a tree from "root" node by uniform sampling
from the set of spanning trees by Wilson's algorithm
            
NOTE: Modifies "graf", adds "tree" attribute to the edges

INPUT: "graf" igraph object
  
OUTPUT: NONE
"""
def wilsonTree(graf, root=0, display=False):
    n = len(graf.vs)
    vertex_seq = range(n)
    
    graf.es["tree"] = False
    graf.vs["marked"] = False
    graf.vs[root]["marked"] = True
    
    
    def loopErase(mylist):
        occur = {}
        for ii in range(len(mylist)):
            occur[mylist[ii]] = ii
            
        outlist = []
        ii = 0
        while (True):
            if ii >= len(mylist):
                return(outlist)
            
            if (occur[mylist[ii]] == ii):
                outlist.append(mylist[ii])
                ii = ii + 1
            else:
                ii = occur[mylist[ii]]
        
    
    def loopErasedRW(start):
        cur_node = start
        node_ls = [cur_node]
        
        while (True):
            cur_node = choices(graf.neighbors(cur_node), k=1)[0]
            node_ls.append(cur_node)
            
            #if (cur_node == start):
            #    node_ls = [cur_node]
                
            if (graf.vs[cur_node]["marked"]):
                
                node_ls = loopErase(node_ls)
                
                graf.vs[node_ls]["marked"] = True
                my_edge_ids = graf.get_eids(path=node_ls)
                
                graf.es[my_edge_ids]["tree"] = True
                return
            
            
    for ii in vertex_seq:
        if (display):
            if (ii % 10000 == 0):
                print((ii, n))
        if graf.vs[ii]["marked"]:
            continue
        else:
            loopErasedRW(ii)
            
    return

"""
INPUT: n -- number of nodes
        m -- total number of edges of the output network
        alpha, beta -- parameters for PA
        K -- number of trees

OUTPUT: [0] is an igraph object
        [1] is n--dim vector, output[1][i] is the root (tree) of node i
"""
def createNoisyGraph(n, m, alpha=0, beta=1, K=1):
    
    res = createPATree(n, alpha, beta, K)
    mytree = res[0]
    clust = res[1]
    
    addRandomEdges(mytree, m)
    return((mytree, clust))
        

def addRandomEdges(graf, m):
    
    n = len(graf.vs)
    assert m < n*(n-1)/2
    
    while(len(graf.es) < m):
        m2 = m - len(graf.es)
        
        heads = choices(range(n), k=m2)
        tails = choices(range(n), k=m2)
        
        edgelist = [(heads[j], tails[j]) for j in range(m2) if heads[j] != tails[j]]

        graf.add_edges(edgelist)
        graf.simplify()
        
    return(graf)        

"""
INPUT: n -- number of nodes
        alpha, beta -- parameters for PA
        K -- number of trees

OUTPUT: [0] is an igraph object
        [1] is n--dim vector, output[1][i] is the root (tree) of node i
"""
def createPATree(n, alpha=0, beta=1, K=1):
    mytree = Graph()
    
    mytree.add_vertices(n)
    edge_ls = []
    
    ## parent of every node; -1 if root
    ## tree ID of every node
    pa_vec = [-1] * n
    tree_vec = [0] * n 
    
    if (K == 1):
        mytree.add_edges( [(0,1)] )
        wt_ls = [alpha+beta, alpha+beta]
        pa_vec[1] = 0
        initi = 2
    else :
        wt_ls = [alpha + 2*beta] * K
        initi = K
        for k in range(K):
            tree_vec[k] = k
    
    for i in range(initi, n):
        
        if (alpha == 1 and beta ==0):
            cur_node = choices(range(i))[0]
        else:
            cur_node = choices(range(i), weights=wt_ls)[0]
        
        wt_ls.append(alpha+beta)
        wt_ls[cur_node] = wt_ls[cur_node] + beta        
        edge_ls.append((cur_node, i))
        
        pa_vec[i] = cur_node
        
        while (pa_vec[cur_node] != -1):
            cur_node = pa_vec[cur_node]
        
        tree_vec[i] = cur_node
        
        
    mytree.add_edges(edge_ls)
    return((mytree, tree_vec))

"""        
REQUIRE: graf must have edge attribute "tree" 

INPUT: root is a vertex ID; mytree.vs[root] should return vertex object
OUTPUT: N/A

EFFECT: creates node attribute "subtree_size" on graf
       creates node attribute "pa" on graf
"""
def countSubtreeSizes(graf, root, prev=None):
    n = len(graf.vs)
    istree = len(graf.es) == (n-1)

    graf.vs[root]["pa"] = prev
    
    counter = 1
    
    edge_ixs = graf.incident(root)
    
    for eid in edge_ixs:
        my_e = graf.es[eid]
        if (not istree) and (not my_e["tree"]):
            continue
        
        next_node = otherNode(my_e, root)
            
        if (next_node == prev):
            continue
        else:
            counter = counter + countSubtreeSizes(graf, next_node, root)
        
    graf.vs[root]["subtree_size"] = counter
    return(counter)


"""
REQUIRE: graf has node attribute "subtree_size"
         graf has node attribute "pa"


INPUT: "graf" -- igraph object. 
       "root" node as an integer. Unimportant
       
OUTPUT: three-tuple. First: n-dim nparray of probabilities
              Second: denominator -- number of all histories
              Third: largest hist(u,t) value, over u, in log-scale
"""

def countAllHist(graf, root):
    n = len(graf.vs)
    
    countSubtreeSizes(graf, root)
    hist = [0] * n
    
    ntree = graf.vs[root]["subtree_size"]
    
    S = collections.deque([root]) ## queue of nodes to visit

    hist[root] = 0
    tree_nodes = []
    
    while (len(S) > 0):
        cur_node = S.popleft()
        tree_nodes.append(cur_node)
        
        hist[root] = hist[root] - np.log(graf.vs[cur_node]["subtree_size"])

        node_ixs = graf.neighbors(cur_node)
        for next_node in node_ixs:
            
            if (graf.vs[next_node]["pa"] != cur_node):
                continue
            S.append(next_node)
            
    S = collections.deque([root]) ## queue of nodes to visit
    
    while (len(S) > 0):
        cur_node = S.popleft()
    
        node_ixs = graf.neighbors(cur_node)
        
        for next_node in node_ixs:
            
            if (graf.vs[next_node]["pa"] != cur_node):
                continue
            
            S.append(next_node)
            
            hist[next_node] = hist[cur_node] + \
                    np.log(graf.vs[next_node]["subtree_size"] /  \
                           (ntree - graf.vs[next_node]["subtree_size"]))
            
            
    loghist = np.array(hist)
    
    thist = np.array([0] * n, dtype=float)

    thist[tree_nodes] = np.exp(loghist[tree_nodes] - max(loghist[tree_nodes]))

    return((thist/np.sum(thist), np.sum(thist), max(loghist)))
        


"""
INPUT: "v_ls" is a list of nodes
OUTPUT: sub-vector of nodes in the tree
    
Require: edges of "graf" has a "tree" attribute
"""
def treeDFS(graf, start, v_ls=None):
    
    stak = [start]
    
    visited = {}
    
    while (len(stak) > 0):
        cur_v = stak.pop(-1)
        visited[cur_v] = 1
        tree_edges = [e for e in graf.incident(cur_v) if graf.es[e]["tree"]]
        tree_nbs = [otherNode(graf.es[e], cur_v) for e in tree_edges]
    
        for u in tree_nbs:
            if u not in visited:
                stak.append(u)
    if (v_ls != None):
        tmp = [v for v in v_ls if v in visited]
    else:
        tmp = list(visited.keys())
        
    return(tmp)


"""
Get tree sizes


"""

def getTreeSizes(graf, tree2root):
    n = len(graf.vs)
    all_sizes = []
    for k in range(len(tree2root)):
        cur_tree = treeDFS(graf, tree2root[k], range(n))
        all_sizes.append(len(cur_tree))
    
    assert sum(np.array(all_sizes)) == n
    
    return(all_sizes)

    