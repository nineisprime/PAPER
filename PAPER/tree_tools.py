#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:10:19 2020

@author: minx


important invariants in the code:
graf.vs has attribute "pa", "subtree_size"
graf.es has attribute "tree"

function "wilsonTree(graf)" adds "tree" attribute
function "countSubtreeSizes(graf, root)" require "tree" attribute 
      on the edges and creates "subtree_size" and "pa"
"""

import time
from random import choices
from igraph import *
import numpy as np
import collections
import scipy.optimize



def getAllTreeDeg(graf):
    """
    Computes tree degree of all nodes in input graph.

    Parameters
    ----------
    graf : igraph object
        Graph with tree attribute on edges.

    Returns
    -------
    np array of the tree degrees of all nodes.

    """
    n = len(graf.vs)
    degs = [0] * n
    for mye in graf.es:
        if (mye["tree"]):
            degs[mye.source] = degs[mye.source] + 1
            degs[mye.target ] = degs[mye.target] + 1
    degs = np.array(degs)
    return(degs)




def treeDegree(graf, v):
    """
    Computes tree degree of a single node v of the input graph

    Parameters
    ----------
    graf : igraph object
        Graph with tree attribute on edges.
    v : int
        node id.

    Returns
    -------
    tree degree of v as a single integer.

    """
    
    edge_ixs = graf.incident(v)
    
    deg = sum([e["tree"] for e in graf.es[edge_ixs]])
    
    return(deg)




def getAncestors(graf, utilde, u = None):
    """
    Returns ancestors utilde, avoiding u.

    Parameters
    ----------
    graf : igraph object
        Input graph.
    utilde : int
        node id.
    u : int, optional
        node id. The default is None.

    Returns
    -------
    -1 if parents of utilde trace to u, 
          otherwise return list of parents of utilde, including utilde, including the root.

    """
    
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
 
        
 
    
def otherNode(my_edge, node_ix):
    """
    Gives other node id of a given edge and one endpoint id

    Parameters
    ----------
    my_edge : igraph edge object
        Input edge.
    node_ix : int
        One endpoint id.

    Returns
    -------
    Node id of the other endpoint.

    """
    if (my_edge.source == node_ix):
        return(my_edge.target)
    else:
        return(my_edge.source)      



def bfsTree(graf, root=0):
    """
    Creates a tree from a given node by breadth-first-search

    Parameters
    ----------
    graf : igraph object
        Input graph.
    root : int, optional
        Start node id. The default is 0.

    Returns
    -------
    None.

    """
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
                

def wilsonTree(graf, root=0, display=False):
    """
    Creates a tree from a given node by uniform sampling
    from the set of spanning trees by Wilson's algorithm. 
    The starting node can be arbitrary. 

    Adds "tree" attribute to input graph edges.

    Parameters
    ----------
    graf : igraph object
        Input graph.
    root : int, optional
        Start node id; Does not affect output tree. The default is 0.
    display : boolean, optional
        Display details. The default is False.

    Returns
    -------
    None.

    """
    
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


def createNoisyGraph(n, m, alpha=0, beta=1, K=1):
    """
    Generates a new graph from PAPER(alpha, beta, K, theta) model
    with n nodes and m edges. 
    Note: theta parameter not used since m is fixed.

    Parameters
    ----------
    n : int
        Num nodes.
    m : int
        Num edges.
    alpha : float, optional
        Parameter. The default is 0.
    beta : float, optional
        Parameter. The default is 1.
    K : int, optional
        Num of clusters. The default is 1.

    Returns
    -------
    0. igraph object
    1. list representation of the underlying tree
       where i-th element is the root of node i

    """
    
    res = createPATree(n, alpha, beta, K)
    mytree = res[0]
    clust = res[1]
    
    addRandomEdges(mytree, m)
    return((mytree, clust))
        

def addRandomEdges(graf, m):
    """
    Add random Erdos--Renyi edges to input graph
    until the graph has m edges.

    Parameters
    ----------
    graf : igraph object
        Input graph.
    m : int
        Final num of edges.

    Returns
    -------
    igraph object.

    """
    
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


def createPATree(n, alpha=0, beta=1, K=1):
    """
    Generates an APA(alpha, beta, K) forest.
    Default parameter is LPA.

    Parameters
    ----------
    n : int
        Num nodes.
    alpha : float, optional
        Parameter. The default is 0.
    beta : float, optional
        Parameter. The default is 1.
    K : int, optional
        Num of component trees. The default is 1.

    Returns
    -------
    0. igraph object.
    1. list representation of the underlying forest
       where i-th element is the root of node i

    """
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



def countSubtreeSizes(graf, root, prev=None):
    """
    Creates new node attribute "subtree_size" and "pa"
    giving the subtree sizes and parent of each node viewing the 
    input tree as being rooted at a given node.
    
    Require: input graph edges have "tree" attribute.

    Parameters
    ----------
    graf : igraph object
        Input graph; creates node attribute "subtree_size" on graf
        creates node attribute "pa" on graf
    root : int
        Node id of root.
    prev : int, optional
        Internal variable used for recursion. The default is None.

    Returns
    -------
    None.

    """
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



def countAllHist(graf, root):
    """
    Computes posterior root probs for a given tree. 
    Require: graf.es has "tree" attribute; graf.vs 
            has "subtree_size" and "pa" attributes.

    Parameters
    ----------
    graf : igraph object
        Input graph.
    root : int
        Root node id. Computes posterior root prob
        for the tree containing root. 

    Returns
    -------
    0. nparray of posterior root probs
    1. cardinality of all histories of input tree
    2. largest h(u,t) value in log-scale

    """
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
        


def treeDFS(graf, start, v_ls=None):
    """
    Require: edges of "graf" has a "tree" attribute

    Parameters
    ----------
    graf : igraph object
        Input graph.
    start : int
        Start node id.
    v_ls : list, optional
        List of nodes. The default is None.

    Returns
    -------
    0. sublist of v_ls of all nodes in the tree
       containg start node. If v_ls==None, returns
       list of all nodes of tree containing start node.

    """
    
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




def getTreeSizes(graf, tree2root):
    """

    Parameters
    ----------
    graf : igraph object
        Input graph.
    tree2root : list
        Root node of all trees.

    Returns
    -------
    0. list of sizes of all trees

    """
    n = len(graf.vs)
    all_sizes = []
    for k in range(len(tree2root)):
        cur_tree = treeDFS(graf, tree2root[k], range(n))
        all_sizes.append(len(cur_tree))
    
    assert sum(np.array(all_sizes)) == n
    
    return(all_sizes)

    