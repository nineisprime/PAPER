#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:44:30 2021

@author: minx
"""

from PAPER.tree_tools import *
import PAPER.gibbsSampling as gibbsSampling

from igraph import *
import numpy as np
from random import choices
from PAPER.estimateAlpha import *



def gibbsGraft(graf, Burn=30, M=50, gap=1, alpha=0, beta=1, K=1,
               display=True, size_thresh=0.01, birth_thresh=.8,
               initroots=None):
    """
    Collapsed grafting sampler for the fixed K setting.

    Parameters
    ----------
    graf : igraph object
        Input graph.
    Burn : int, optional
        Num of burn in iterations. The default is 30.
    M : int, optional
        Num of regular iterations. The default is 50.
    gap : int, optional
        Num of samples to skip when recording results. 
        The default is 1.
    alpha : float, optional
        Parameter. The default is 0.
    beta : float, optional
        Parameter. The default is 1.
    K : int, optional
        Num of roots/clusters. The default is 1.
    display : boolean, optional
        Detailed display. The default is True.
    size_thresh : float, optional
        Thresh for keeping a cluster-tree. The default is 0.01.
    birth_thresh : float, optional
        Thresh for creating new distinct cluster-tree 
        in output. The default is 0.8.
    initroots : list, optional
        Initialization for set of roots. The default is None.

    Returns
    -------
    0. nparray of length n of posterior root prob
    1. dict giving posterior root prob for each distinct cluster-tree
    2. nparray matrix of node-tree co-occurrences
    3. final set of roots (used for initialization)

    """
    
    
    n = len(graf.vs)
    
    if (initroots is None):
        wilsonTree(graf)
        v = choices(range(n))[0]
    
        countSubtreeSizes(graf, v)
    
        tree2root = [v]

        ## cuts wilson tree into K pieces in an easy way    
        if (K > 1):
            initpi = gibbsSampling.sampleOrdering(graf, tree2root, alpha, beta)
            gibbsSampling.nodewiseSamplePA(graf, initpi, alpha, beta, K)
            tree2root = initpi[0:K]

    else:
        tree2root = initroots
    
    sizes = getTreeSizes(graf, tree2root)
    
    for k in range(K):
        countSubtreeSizes(graf, tree2root[k])
    
    node_tree_coo = np.zeros((n, 0))
    freq = {}
    if (K == 1):
        freq[0] = [0] * n
        bigK = 1
    else:   
        bigK = 0
    
    for i in range(Burn + M):
        treedegs = getAllTreeDeg(graf)  
        assert sum(treedegs) == 2*(n-K)
        
        for k in range(K):
            tree2root[k] = sampleRoot(graf, tree2root[k], alpha, beta, 
                                      single_root=(K==1), degs=treedegs)
            countSubtreeSizes(graf, tree2root[k])
        
        
        for ii in range(n):
            if (ii in tree2root):
                continue
           
            graftSubtreePA(graf, ii, tree2root, alpha, beta, degs=treedegs)

        ## Display and debug
        sizes = getTreeSizes(graf, tree2root)
        sizes_sorted = -np.sort( - np.array(sizes))
        sizes_args = np.argsort(- np.array(sizes) ) 
        
        if (display):
            print("iter {0}   sizes {1}".format(i, sizes_sorted))    
            for v in range(n):
                if (v in tree2root):
                    assert graf.vs[v]["pa"] == None
                else:
                    assert graf.vs[v]["pa"] != None    
        
        tree2root_sorted = [0] * len(tree2root)       
        for k in range(len(tree2root)):
            tree2root_sorted[k] = tree2root[sizes_args[k]]
            
        
        """ record results """
        
        if (i >= Burn and i % gap ==0):
            if (K == 1):
                freq[0] = freq[0] + countAllHist(graf, tree2root[0])[0]
            else:   

                node_tree_coo = gibbsSampling.updateInferResults(graf, freq, tree2root,
                                                   alpha=alpha, beta=beta, size_thresh=size_thresh,
                                                   birth_thresh=birth_thresh, 
                                                   node_tree_coo=node_tree_coo)
            
    allfreqs = np.array([0] * n)

    for k in range(len(freq)):
        allfreqs = allfreqs + freq[k]
        freq[k] = freq[k]/sum(freq[k])
     
    allfreqs = allfreqs/sum(allfreqs)
    return((allfreqs, freq, node_tree_coo, tree2root))




def gibbsGraftDP(graf, Burn=30, M=50, gap=1, alpha=0, beta=1, alpha0=5, 
                 display=True, size_thresh=0.01, 
                 birth_thresh=0.8, initroots=None):
    """
    Collapsed grafting sampler for the random K setting.

    Parameters
    ----------
    graf : igraph object
        Input graph.
    Burn : int, optional
        Num of burn in iterations. The default is 30.
    M : int, optional
        Num of regular iterations. The default is 50.
    gap : int, optional
        Num of samples to skip when recording results. 
        The default is 1.
    alpha : float, optional
        Parameter. The default is 0.
    beta : float, optional
        Parameter. The default is 1.
    alpha0 : float, optional
        Parameter. The default is 5.
    display : boolean, optional
        Detailed display. The default is True.
    size_thresh : float, optional
        Thresh for keeping a cluster-tree. The default is 0.01.
    birth_thresh : float, optional
        Thresh for creating new distinct cluster-tree 
        in output. The default is 0.8.
    initroots : list, optional
        root initialization. The default is None.

    Returns
    -------
    0. nparray of length n of posterior root prob
    1. dict giving posterior root prob for each distinct cluster-tree
    2. list of all Ks
    3. final alpha0 (used for initialization)
    4. final set of roots (used for initialization)

    """
    
    n = len(graf.vs)
    display=True
    if (initroots is None):
        wilsonTree(graf)
        v = choices(range(n))[0]
        
        countSubtreeSizes(graf, v)
    
        tree2root = [v]
    else:
        tree2root = initroots
    
  
    allK = []
   
    
    if (display):
        print("Starting gibbsGraftDP ...")
    
    freq = {}
    bigK = 0
    
    
    for i in range(Burn + M):
        
        treedegs = getAllTreeDeg(graf)
        
        for k in range(len(tree2root)):
            tree2root[k] = sampleRoot(graf, tree2root[k], alpha, beta, degs=treedegs)
            countSubtreeSizes(graf, tree2root[k])
        
        
        for ii in range(n):
            if (ii in tree2root and len(tree2root) == 1):
                continue
            tree2root = graftSubtreeDP(graf, ii, tree2root, alpha, beta, alpha0, 
                                       degs=treedegs)[1]
        
        K = len(tree2root)
        alpha0tilde = drawAlpha0tilde(K, n, alpha0/(alpha+2*beta))
        alpha0 = alpha0tilde*(alpha+2*beta)    
    
        sizes = getTreeSizes(graf, tree2root)
        sizes_sorted = -np.sort( - np.array(sizes))
        sizes_args = np.argsort( - np.array(sizes))
    
        if (display):
            print("iter {0}  a0 {1}  K {2}  sizes {3}".format(i, round(alpha0, 3), 
                                                          K, sizes_sorted[0:6]))    
        for v in range(n):
            if (v in tree2root):
                assert graf.vs[v]["pa"] == None
            else:
                assert graf.vs[v]["pa"] != None    
        
        
        """ record results:
        update variable freq, bigK, allK    
        """
        if (i >= Burn and i % gap == 0):
            
            allK.append(len(tree2root))
            
            gibbsSampling.updateInferResults(graf, freq, tree2root,
                               alpha=alpha, beta=beta,
                               size_thresh=size_thresh,
                               birth_thresh=birth_thresh)   

    
    
    allfreqs = np.array([0] * n)        
    for k in range(len(freq)):
        allfreqs = allfreqs + freq[k]
        freq[k] = freq[k]/sum(freq[k])
    
    allfreqs = allfreqs/sum(allfreqs)
    return((allfreqs, freq, allK, alpha0, tree2root))


    
def sampleRoot(graf, v, alpha, beta, single_root=False, degs=None):
    """
    Generates a new root for the single tree containing node v.

    Parameters
    ----------
    graf : igraph object
        Input graph.
    v : int
        Node id representing the tree.
    alpha : float
        Parameter.
    beta : float
        Parameter.
    single_root : boolean, optional
        True iff K==1. The default is False.
    degs : list, optional
        list of all the tree degrees. The default is None.

    Returns
    -------
    0. index of new root

    """
    n = len(graf.vs)
    
    if (graf.vs[v]["subtree_size"] == 1):
        return(v)
    
    normalized_h = countAllHist(graf, v)[0]
    
    if (degs is None):
        degs = getAllTreeDeg(graf)
    
    deg_adj = (beta*degs + beta + alpha)*(beta*degs + alpha)
    
    if (not single_root):
        tmp_p = normalized_h*deg_adj
    else:
        tmp_p = normalized_h
    
    
    newroot = choices(range(n), tmp_p)[0]
    return(newroot)
    




def graftSubtreeDP(graf, u, tree2root, alpha, beta, alpha0, degs=None):
    """
    Generates a new parent for node u, potentially 
    making u a new root node. Only for the random K setting.

    Parameters
    ----------
    graf : igraph object
        Input graph.
    u : int
        Current node id.
    tree2root : list
        list of all roots.
    alpha : float
        Parameter. 
    beta : float
        Parameter. 
    alpha0 : float
        Parameter
    degs : list, optional
        List of all tree degrees. The default is None.

    Returns
    -------
    0. node index of new parent of u; -1 if u becomes a new root
    1. new list of roots

    """
    n = len(graf.vs)
    m = len(graf.es)
    n2 = n*(n-1)/2
    K = len(tree2root)
    
    u_edge_ixs = graf.incident(u)
    old_pa = graf.vs[u]["pa"]
    uisroot = (old_pa == None)
    
    u_size = graf.vs[u]["subtree_size"]
    
    wts = []
    pas = []   
    
    for eid in u_edge_ixs:
        my_e = graf.es[eid]
        
        utilde = otherNode(my_e, u)
        
        ants = getAncestors(graf, utilde, u)
        if (ants == -1):
            continue
        
        if (degs is None):
            utildedeg = treeDegree(graf, utilde)
        else:
            utildedeg = degs[utilde]

        if (old_pa == utilde):
            ants_big_sizes = np.array(graf.vs[ants]["subtree_size"])
            ants_sm_sizes = ants_big_sizes - u_size
            if (utilde in tree2root):
                deg_adj = beta*(utildedeg+1) + alpha
            else:
                deg_adj = beta*(utildedeg-1) + alpha
        
        else:
            ants_sm_sizes = np.array(graf.vs[ants]["subtree_size"])
            ants_big_sizes = ants_sm_sizes + u_size
            if (utilde in tree2root):
                deg_adj = beta*(utildedeg+2) + alpha
            else:
                deg_adj = beta*(utildedeg) + alpha
        
        my_wt = deg_adj * \
            np.exp(np.sum(np.log(ants_sm_sizes) - np.log(ants_big_sizes)))
            
        pas.append(utilde)
        wts.append(my_wt)
    
    """ add option for u to become root """
    my_wt = alpha0 * (m-n+K+1 -uisroot)/(n2-n+K+1 -uisroot)
        
    if (degs is None):
        udeg = treeDegree(graf, u)
    else:
        udeg = degs[u]
    
    my_wt = my_wt * (beta*udeg +beta*uisroot +alpha)/(beta+alpha) 
    
    wts.append(my_wt)
    pas.append(-1)
    
    """ choose the new parent """
    new_pa = choices(pas, wts)[0]
    
    if (new_pa == -1): 
        new_pa = None
    
    """ make adjustments """    
    if (new_pa == old_pa):
        return((old_pa, tree2root))
    else:
        
        graf.vs[u]["pa"] = new_pa
        
        if (old_pa != None):
            old_edge_ix = graf.get_eids( [(u, old_pa)] )[0]
            graf.es[old_edge_ix]["tree"] = 0
        
            old_ants = getAncestors(graf, old_pa, u)
            for w in old_ants:
                graf.vs[w]["subtree_size"] = graf.vs[w]["subtree_size"] - u_size
                assert graf.vs[w]["subtree_size"] > 0
        else:
            tree2root.remove(u)
            
        if (new_pa != None):
            new_edge_ix = graf.get_eids( [(u, new_pa)] )[0]
            graf.es[new_edge_ix]["tree"] = 1
        
            new_ants = getAncestors(graf, new_pa, u)
            for w in new_ants:
                graf.vs[w]["subtree_size"] = graf.vs[w]["subtree_size"] + u_size
        else:
            tree2root.append(u)
        
        if (degs is not None):
            if (old_pa != None):
                degs[old_pa] = degs[old_pa] - 1
            else:
                degs[u] = degs[u] + 1
                
            if (new_pa != None):
                degs[new_pa] = degs[new_pa] + 1
            else:
                degs[u] = degs[u] - 1
        
        return((new_pa, tree2root))
        
    
    
def graftSubtreePA(graf, u, tree2root, alpha, beta, degs=None):
    """
    Generates a new parent for node u. 
    Only for the fixed K setting

    Parameters
    ----------
    graf : igraph object
        Input graph.
    u : int
        Current node id.
    tree2root : list
        list of all roots.
    alpha : float
        Parameter. 
    beta : float
        Parameter. 
    degs : list, optional
        List of all tree degrees. The default is None.

    Returns
    -------
    node index of new parent of u; -1 if u becomes a new root

    """
    assert not (u in tree2root)

    multi_root = (len(tree2root) > 1)
    
    u_size = graf.vs[u]["subtree_size"]
    
    pas = []
    wts = []
    u_edge_ixs = graf.incident(u)
    
    old_pa = graf.vs[u]["pa"]
    
    if (old_pa == None):
        print("culprit {0}".format(u))
    assert old_pa != None
    
    for eid in u_edge_ixs:
        my_e = graf.es[eid]
        utilde = otherNode(my_e, u)
        
        ants = getAncestors(graf, utilde, u)
        if (ants == -1):
            continue

        if (degs is None):
            utildedeg = treeDegree(graf, utilde)
        else:
            utildedeg = degs[utilde]
        
        """ if utilde is root, do not compute subtree sizes"""
        if (len(ants) == 1):
            assert graf.vs[utilde]["pa"] == None
            my_wt = 1
            
            deg_adj = beta*(utildedeg-(utilde == old_pa)+ \
                             2*multi_root) + alpha
            
            wts.append(my_wt*deg_adj)
            pas.append(utilde)
            continue
        
        """ if utilde is not root, compute subtree size without
            root node """
        ants.pop(-1)
        if (old_pa == utilde):
            ants_big_sizes = np.array(graf.vs[ants]["subtree_size"])
            ants_sm_sizes = ants_big_sizes - u_size  
            deg_adj = beta*(utildedeg-1) + alpha
            
        else:
            ants_sm_sizes = np.array(graf.vs[ants]["subtree_size"])
            ants_big_sizes = ants_sm_sizes + u_size
            deg_adj = beta*utildedeg + alpha
    
        my_wt = deg_adj * \
            np.exp(np.sum(np.log(ants_sm_sizes) - np.log(ants_big_sizes)))
    
        pas.append(utilde)
        wts.append(my_wt)
        
    """ draw a new parent """
    new_pa = choices(pas, weights=wts)[0]
    
    if (new_pa == old_pa):
        return(old_pa)
    else:
        graf.vs[u]["pa"] = new_pa
        old_edge_ix = graf.get_eids( [(u, old_pa)] )[0]
        graf.es[old_edge_ix]["tree"] = False
    
        new_edge_ix = graf.get_eids( [(u, new_pa)] )[0]
        graf.es[new_edge_ix]["tree"] = True
    
        if (degs is not None):
            degs[old_pa] = degs[old_pa] - 1
            degs[new_pa] = degs[new_pa] + 1
    
    
        new_ants = getAncestors(graf, new_pa, u)
        for ix in new_ants:
            graf.vs[ix]["subtree_size"] = graf.vs[ix]["subtree_size"] + u_size
                
        
        old_ants = getAncestors(graf, old_pa, u)
        for ix in old_ants:
            graf.vs[ix]["subtree_size"] = graf.vs[ix]["subtree_size"] - u_size
            assert graf.vs[ix]["subtree_size"] > 0
            
        return(new_pa)        
        
    
    
    
    
    
    
    
    
    
    
