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
    degs = np.array([0] * n)
    
    for i in range(n):
        mypa = graf.vs[i]["pa"]
        if (mypa != None):
            degs[mypa] = degs[mypa] + 1
            degs[i] = degs[i] + 1
    
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



def createSeqGraph(n, alpha=0, beta=1, K=1, theta=1, talpha=0, tbeta=1,
                   eta=0, gif = False, gifname = "def"):
    """
    ...

    """
    mygraph = Graph()
    
    mygraph.add_vertices(n)
    mygraph.vs["label"] = range(n)

    pa_vec = [-1] * n           ## tree parent of every node; -1 if root
    tree_vec = [0] * n          ## tree ID of every node
    filenames = []
    
    if (K == 1):
        mygraph.add_edges( [(0,1)])
        mygraph.es["type"] = "tree"
        deg_ls = [1, 1]
        pa_vec[1] = 0
        initi = 2
    else :
        deg_ls = [2] * K
        initi = K
        for k in range(K):
            tree_vec[k] = k
    
    for i in range(initi, n):
        
        # Recruitment Stage (Forest construction)
        if (alpha == 1 and beta ==0):
            cur_node = choices(range(i))[0]
        else:
            wt_ls = [beta*deg_ls[j] + alpha for j in range(i)]
            cur_node = choices(range(i), weights=wt_ls)[0]

        # print("parent node chosen: {}".format(cur_node))
        
        deg_ls.append(1)
        deg_ls[cur_node] = deg_ls[cur_node] + 1        
        tree_edge_ls=[(cur_node, i)]
        
        pa_vec[i] = cur_node
        
        # Connection Stage (Adding noise)
        noise_probs = [theta*(tbeta*deg_ls[j]+talpha)/(2*(i-1)*tbeta + i*talpha) for j in range(i)]
        
        ebool = np.random.uniform(size=i) < np.array(noise_probs)
        node_ls = [x for x in range(i)]
        
        noise_idx = [a for a,b in zip(node_ls,ebool) if b != 0 and a != cur_node]    
        noise_edge_ls = []


        for noise in noise_idx:
            noise_edge_ls.append((noise, i))
        
        # Retrace to obtain the community root
        
        while (pa_vec[cur_node] != -1):
            cur_node = pa_vec[cur_node]
        
        tree_vec[i] = cur_node
        
        mygraph.add_edges(tree_edge_ls)
        mygraph.es[-1]["type"] = "tree"
        mygraph.add_edges(noise_edge_ls)
        
        if (len(noise_edge_ls) > 0):
            mygraph.es[-len(noise_edge_ls):]["type"]="noise"
    
    
    tree_bools = choices([1, 0], weights=[eta, 1-eta], k=n-K)
    tree_edges = [e for e in mygraph.es if e["type"] == "tree"]
    assert len(tree_edges) == n-K
    mygraph.delete_edges([tree_edges[j] for j in range(n-K) if tree_bools[j]])
    

    
    
    
    if (gif):
    
        mygraph2 = mygraph.copy()
        mygraph2.delete_edges([mye for mye in mygraph2.es if mye["type"] != "tree"])
    
        mylayout = mygraph2.layout_fruchterman_reingold(niter=700)
        
        
        mygraph2.vs["seq"]="unvisited"

        color_dict = {"tree": "rgba(60,60,60,0.8)", 
                      "curr":"red", 
                      "noise": "pink",
                      "visited":"black",
                      "unvisited" : "white",
                      "curr":"red"}

        size_dict = {"visited" : 800/n,
                     "curr" : 800/n,
                     "unvisited" : 0}

        mygraph2.delete_edges(mygraph2.es)
        mygraph2.vs["label"] = None
        
        mygraph2.vs[0]["seq"] = "visited"
        mygraph2.vs[1]["seq"] = "visited"

        if (not tree_bools[0]):
            mygraph2.add_edges( [(0,1)] )
            mygraph2.es[-1]["type"] = "tree"

        for i in range(initi, n):
            
            cur_node = pa_vec[i]
            
            mygraph2.vs[i]["seq"] = "visited"
            mygraph2.vs[cur_node]["seq"] = "curr"
            
            mygraph2.vs["color"] = [color_dict[v] for v in mygraph2.vs["seq"]]
            mygraph2.vs["size"] = [size_dict[v] for v in mygraph2.vs["seq"]]
            
            noise_edges = [(otherNode(e, i), i) for e in mygraph.es[mygraph.incident(i)] if \
                         otherNode(e, i) < i and otherNode(e, i) != cur_node]
                
            mygraph2.add_edges(noise_edges)
            
            if (len(noise_edges) > 0):
                mygraph2.es[-len(noise_edges):]["type"] = "noise"
            
            if (not tree_bools[i-1]):
                mygraph2.add_edges( [(i, cur_node)] )
                mygraph2.es[-1]["type"] = "tree"
            
            mygraph2.es["color"] = [color_dict[e] for e in mygraph2.es["type"]]
            
            f = "{}_{}.png".format(gifname,i)

            plot(mygraph2, f, layout=mylayout, margin=20)

            for i in range(2):
                filenames.append(f)
        
            mygraph2.vs[cur_node]["seq"] = "visited"
            
        

    if(gif):
    
        # build gif
        with imageio.get_writer('{}.gif'.format(gifname), mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
            
        # Remove files
        for filename in set(filenames):
           os.remove(filename)    
            
    
    return((mygraph, tree_vec))    




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



"""
Assumes "root" is in nodes. Sets root["pa"] = None and
adjusts "pa" and "subtree_size" attributes of all v in "nodes"

"subtrees_size" is computed for with respect to the subtree induced 
by the subset of "nodes"

INPUT: graf
       
NOTE: modifies "pa" and "subtree_size" attribute on node in "nodes"

"""
def adjustSubtreeSizes(graf, nodes, root):    
    return adjustSubtreeSizes_helper(graf, nodes, cur_node=root)

def adjustSubtreeSizes_helper(graf, nodes, cur_node, prev=None):

    # BFS helper

    n = len(graf.vs)
    istree = len(graf.es) == (n-1)

    graf.vs[cur_node]["pa"] = prev                      # parent node
    
    if cur_node not in nodes:
        #counter = graf.vs[cur_node]["subtree_size"]
        #return counter
        return 0
    
    counter = 1
    
    edge_ixs = graf.incident(cur_node)                  # edge list connected to cur_node
    
    for eid in edge_ixs:
        my_e = graf.es[eid]
        if (not istree) and (not my_e["tree"]):
            continue                                    # skip non-tree edges 
        
        next_node = otherNode(my_e, cur_node)           
            
        if (next_node == prev):                         # skip pa
            continue
        else:
            counter = counter + adjustSubtreeSizes_helper(graf, nodes, next_node, cur_node)
        
    graf.vs[cur_node]["subtree_size"] = counter
    return(counter)




def getUnmarkedAncestor(graf, utilde, marked):
        
    cur_node = utilde
    
    while (True):
        my_pa = graf.vs[cur_node]["pa"]
        
        if my_pa in marked:
            return cur_node
        else:
            cur_node = my_pa




def pastDegree(graf, v, j, pi, pi_inv):
    """
    Return D_{j-1}(v) with respect to tree.

    Parameters
    ----------
    graf : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    j : TYPE
        DESCRIPTION.
    pi : TYPE
        DESCRIPTION.
    pi_inv : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    assert j > pi_inv[v]
    
    nb_edges_v = graf.es[graf.incident(v)]
    nb_tree_edges_v = [e for e in nb_edges_v if e["tree"]]
    tree_nbs_v = [otherNode(e, v) for e in nb_tree_edges_v]
        
    Dv = len([w for w in tree_nbs_v if pi_inv[w] < j])
    
    if (graf.vs[v]["pa"] is not None and graf.vs[v]["pa"] not in tree_nbs_v):
        Dv = Dv + 1
    
    return(Dv)







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
        



"""
Computes hist(u, subtree) for all u in "perm" with
respect to the subtree induced by the subset "perm"

REQUIRE: graf has node attribute "subtree_size"
         graf has node attribute "pa"
         perm is a substring of pi and must contain 0
        

INPUT: "graf" -- igraph object. 
       "perm" -- vertices to compute the probabilities for 
       "root" node as an integer. Unimportant. Has to be in perm
       
OUTPUT: hist, n0-dim nparray of probabilities where n0 = len(perm)
        hist[i] = hist(subpi[i], subtree)
"""
def countSubtreeHist(graf, perm, root):
    #n = len(graf.vs)
    
    n0 = len(perm)
    perm_inv = {}
    
    for i in range(n0):
        perm_inv[perm[i]] = i
        
    
    adjustSubtreeSizes(graf, perm, root)
    hist = [0] * n0
    
    ntree = graf.vs[root]["subtree_size"]
    
    S = collections.deque([root]) ## queue of nodes to visit

    hist[perm_inv[root]] = 0
    tree_nodes = []
    
    while (len(S) > 0):                                                 
        cur_node = S.popleft()
 
        tree_nodes.append(cur_node)
        
        hist[perm_inv[root]] = hist[perm_inv[root]] - \
                np.log(graf.vs[cur_node]["subtree_size"])               

        node_ixs = graf.neighbors(cur_node)
        
        for next_node in node_ixs:
            if next_node not in perm_inv:
                continue
            
            if (graf.vs[next_node]["pa"] != cur_node):
                continue
            
            S.append(next_node)                                         # Search neighbors for child node
            
    S = collections.deque([root]) ## queue of nodes to visit
    
    while (len(S) > 0):
        cur_node = S.popleft()
    
        node_ixs = graf.neighbors(cur_node)
        
        for next_node in node_ixs:
            if next_node not in perm_inv:
                continue
            
            if (graf.vs[next_node]["pa"] != cur_node):
                continue
            
            S.append(next_node)
            
            hist[perm_inv[next_node]] = hist[perm_inv[cur_node]] + \
                    np.log(graf.vs[next_node]["subtree_size"] /  \
                           (ntree - graf.vs[next_node]["subtree_size"]))
            
            
    loghist = np.array(hist)
    
    thist = np.array([0] * n0, dtype=float)

    thist = np.exp(loghist - max(loghist))

    return thist/np.sum(thist)





def nonRootSwapConsistent(graf, pi, pi_inv, pair):
    """
    Write pair = (j, k) where j < k

    Return true iff all children of pi[j] has pi-position > k AND
    parent of pi[k] has pi-position < j

    """    
    n = len(graf.vs)
    
    j, k = min(pair), max(pair)
    
    pij_tree_edges = [graf.es[e] for e in graf.incident(pi[j]) if graf.es[e]["tree"]]
    pij_child = [otherNode(e, pi[j]) for e in pij_tree_edges]
    
    if (graf.vs[pi[j]]["pa"] in pij_child):
        pij_child.remove(graf.vs[pi[j]]["pa"])
    
    cons = all([pi_inv[c] > k for c in pij_child])
    
    cons = cons and ( pi_inv[ graf.vs[pi[k]]["pa"] ] < j )

    return(cons)




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

    