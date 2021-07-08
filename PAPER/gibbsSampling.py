#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:43:28 2021

@author: minx
"""

from tree_tools import *
import time
from random import *
from igraph import *
import numpy as np
import collections
import scipy.optimize


import grafting
from estimateAlpha import *




"""
Executes either gibbsTree or gibbsTreeDP to convergence

INPUT: graf, igraph object
        DP, boolean indicating whether to use Dirichlet process
        tol, criterion for convergence in Hellinger distance
        
OUTPUT: [0]  n-dim vector, u -> P(u in S | graf)  where S is set of roots

"""
def gibbsTreeToConv(graf, DP=False, K=1, alpha=0, beta=0, alpha0=60,
                Burn=40, M=60, gap=1, 
                MAXITER=300, tol=0.1, size_thresh=0.01):
    
    n = len(graf.vs)
    m = len(graf.es)
    graf2 = graf.copy()
    
    if (alpha == 0 and beta == 0):
        beta = 1
        alpha = estimateAlphaEM(graf, display=False)
        print("Estimated alpha as {0}".format(alpha))
    
    options = {"Burn": Burn, "M": M, "gap": gap, "alpha": alpha, 
               "beta": beta, "display": False, "size_thresh": size_thresh}    
    if (not DP):
        res = gibbsTree(graf, K=K, initpi=None, **options)
        res1 = gibbsTree(graf2, K=K, initpi=None, **options)
    else:
        res = gibbsTreeDP(graf, alpha0=alpha0, initpi=None, **options)
        res1 = gibbsTreeDP(graf2, alpha0=alpha0, initpi=None, **ptions)        


    allfreq = np.array([0] * n)
    allfreq1 = np.array([0] * n)
    
    for i in range(MAXITER):
        
        allfreq = allfreq + np.array(res[0])
        allfreq1 = allfreq1 + np.array(res1[0])
            
        p1 = allfreq/sum(allfreq)
        p2 = allfreq1/sum(allfreq1)
        
        deviation = (1/2)*sum(np.abs( p1**(1/2) - p2**(1/2) )**2)
        print((i, deviation))
        
        if (deviation < tol):
            break
        
        if (deviation > .9):
            allfreq = np.array([0] * n)
            allfreq1 = np.array([0] * n)
        
        Mp = M*(i+1)
        
        options["Burn"] = 0
        options["M"] = Mp
        
        if (not DP):
            res = gibbsTree(graf, K=K, initpi=res[-1], **options)
            
            res1 = gibbsTree(graf2, K=K, initpi=res1[-1], **options)
        else:
            res = gibbsTreeDP(graf, alpha0=res[-2], 
                              initpi=res[-1], initroots=res[-3],
                              **options)
            
            res1 = gibbsTreeDP(graf2, alpha0=res1[-2],
                               initpi=res1[-1], initroots=res1[-3],
                               **options)       
    
    allfreq = allfreq + allfreq1
    allfreq = allfreq/sum(allfreq)
    
    return((allfreq, res, res1))



"""
INPUT: 

OUTPUT: 0: allfreqs, sum of all freqs normalized
        1: freq, if K==1, then a single vector
                if K>1, then a dictionary mapping k to vectors
        2: node_tree_coo, (n --by-- K) matrix
        3: tree2root
        4: mypi, n--dim vector
"""
def gibbsTree(graf, Burn=40, M=50, gap=1, alpha=0, beta=1, K=1, 
              display=True, size_thresh=0.01, initpi=None):
    
    n = len(graf.vs)
    m = len(graf.es)

    if (initpi is None):
        wilsonTree(graf)
        v = choices(range(n))[0]
    
        countSubtreeSizes(graf, v)
        tree2root = [v]
        initpi = sampleOrdering(graf, tree2root, alpha, beta)
    else:
        tree2root = initpi[0:K]
    
    mypi = initpi
    
    
    node_tree_coo = np.zeros((n, 0))
    
    freq = {}
    if (K == 1):
        freq[0] = [0] * n
        bigK = 1
    else:   
        bigK = 0
    
    for i in range(Burn + M):
        
        for v in tree2root:
            assert graf.vs[v]["pa"] is None        
        
        nodewiseSamplePA(graf, mypi, alpha=alpha, beta=beta, K=K)
        tree2root = mypi[0:K]
        mypi = sampleOrdering(graf, tree2root, alpha=alpha, beta=beta)
        
        
        ## sort and display sizes
        sizes = getTreeSizes(graf, tree2root)
        sizes_sorted = -np.sort( - np.array(sizes))
        sizes_args = np.argsort(- np.array(sizes) )
        
        if (display):
            print("iter {0}  sizes {1}".format(i, sizes_sorted))
        
        tree2root_sorted = [0] * len(tree2root)
        for k in range(len(tree2root)):
            tree2root_sorted[k] = tree2root[sizes_args[k]]
        
        """ record results """
        
        if (i >= Burn and i % gap == 0):
            if (K == 1):
                freq[0] = freq[0] + countAllHist(graf, tree2root[0])[0]
            else:   

                tmp_freq = {}
                
                treedegs = getAllTreeDeg(graf)
                
                kk = 0
                for k in range(K):
                    if (sizes_sorted[k] > size_thresh * n):
                        tmp_freq[k] = countAllHist(graf, tree2root_sorted[k])[0]
                        
                        if (sizes_sorted[k] > 1):
                            tmp_freq[kk] = tmp_freq[kk] * (beta*treedegs+beta+alpha) \
                                * (beta*treedegs + alpha)                           

                            tmp_freq[kk] = tmp_freq[kk]/sum(tmp_freq[kk])
                        
                        kk = kk + 1
                mybigK = kk
 
                if (mybigK > bigK):
                    for k in range(bigK, mybigK):
                        freq[k] = np.array([0] * n)
                        node_tree_coo = np.column_stack((node_tree_coo, np.zeros((n,1))))                        
                    bigK = mybigK
 
               # match each tree in the forest to existing records
                        
                dists = np.zeros((mybigK, bigK))
                    
                for k in range(mybigK):
                    for kk in range(bigK):
                        if (sum(freq[kk]) > 0):
                            distr1 = freq[kk]/sum(freq[kk])
                            distr2 = np.array(tmp_freq[k])
                            dists[k, kk] = sum(np.abs(distr1 - distr2))/2
                        else:
                            dists[k, kk] = 0
                    
                treematch = scipy.optimize.linear_sum_assignment(dists)[1]
                    
                if (display):
                    print([ round(dists[k, treematch[k]], 4) for k in range(mybigK)])
                
                
                for k in range(mybigK):
                    if (dists[k, treematch[k]] > .75):
                        freq[bigK] = np.array([0] * n)
                        treematch[k] = bigK
                        node_tree_coo = np.column_stack((node_tree_coo, np.zeros((n,1))))
                        bigK = bigK+1
                
                for k in range(mybigK):                    
                    freq[treematch[k]] = freq[treematch[k]] + tmp_freq[k]
                
                                
                ## compute co-occurrence between node and tree
                for ii in range(n):
                    ants = getAncestors(graf, ii)
                    myroot = ants[-1]
                    my_k = tree2root_sorted.index(myroot)
                    if (sizes_sorted[my_k] <= size_thresh * n):
                        continue
                    
                    my_kstar = treematch[my_k]
                    node_tree_coo[ii, my_kstar] = node_tree_coo[ii, my_kstar] + 1     

    allfreqs = np.array([0] * n)    
    
    for k in range(bigK):
        allfreqs = allfreqs + freq[k]
        freq[k] = freq[k]/sum(freq[k])
        
    allfreqs = allfreqs/sum(allfreqs)
    
    return((allfreqs, freq, node_tree_coo, tree2root, mypi))
        


"""

OUTPUT: 
    0: allfreqs
    1: allK, list of length M
    2: tree2root
    3: alpha0
    4: mypi
    
"""
def gibbsTreeDP(graf, Burn=20, M=50, gap=1, alpha=0, beta=1, alpha0=50, 
                display=True, size_thresh=0.01, initpi=None, initroots=None):
    
    n = len(graf.vs)
    m = len(graf.es)
    
    if (initpi is None):
        """
        initpi = np.random.permutation(list(range(n)))
        tree2root = list(range(n))
        graf.es["tree"] = 0
        """
        
        wilsonTree(graf)
        v = choices(range(n))[0]
    
        countSubtreeSizes(graf, v)
        tree2root = [v]
    
        tmp = sampleOrdering(graf, tree2root, alpha, beta, DP=True)
        initpi = tmp[0]
        tree2root = tmp[1]
    else:
        tree2root = initroots
        
    mypi = initpi
    
    allK = []
    
    freq = {}
    bigK = 0    
    
    for i in range(Burn + M):
            
        tree2root = nodewiseSampleDP(graf, mypi, tree2root, alpha=alpha, beta=beta, alpha0=alpha0)
        
        sizes = getTreeSizes(graf, tree2root)
                
        tmp = sampleOrdering(graf, tree2root, alpha=alpha, beta=beta, DP=True)
        mypi = tmp[0]
        tree2root = tmp[1]
    
    
        K = len(tree2root)
    
        sizes_sorted = -np.sort( - np.array(sizes))
        sizes_args = np.argsort( - np.array(sizes))
    
        ## Uncomment to update alpha0
        alpha0tilde = drawAlpha0tilde(K, n, alpha0/(alpha+2*beta))
        alpha0 = alpha0tilde*(alpha+2*beta)
                      
        tree2root_sorted = [0] * K
        for k in range(K):
            tree2root_sorted[k] = tree2root[sizes_args[k]]
        
        
        if (display):
            print("iter {0}  a0 {1}  K {2}  sizes{3}".format(i, round(alpha0, 3),
                                                             K, sizes_sorted))
            
        """ record results """

        if (i >= Burn and i % gap == 0):
            allK.append(K)
            
            tmp_freq = {}
            
            treedegs = getAllTreeDeg(graf)
            
            kk = 0
            for k in range(K):
                if (sizes_sorted[k] > size_thresh * n):
                    tmp_freq[kk] = countAllHist(graf, tree2root_sorted[k])[0]
                    
                    if (sizes_sorted[k] > 1):
                        tmp_freq[kk] = tmp_freq[kk] * (beta*treedegs+beta+alpha) \
                                * (beta*treedegs + alpha)                           

                        tmp_freq[kk] = tmp_freq[kk]/sum(tmp_freq[kk])
                    
                    kk = kk + 1
            mybigK = kk
            
            if (mybigK > bigK):
                for k in range(bigK, mybigK):
                    freq[k] = np.array([0] * n)
                bigK = mybigK
            
            dists = np.zeros((mybigK, bigK))
            
            for k in range(mybigK):
                for kk in range(bigK):
                    if (sum(freq[kk] > 0)):
                        
                        distr1 = np.array(freq[kk]/sum(freq[kk]) )                        
                        distr2 = np.array(tmp_freq[k])

                        dists[k, kk] = sum(np.abs(distr1 - distr2))/2
                    else:
                        dists[k, kk] = 0
                    
            treematch = scipy.optimize.linear_sum_assignment(dists)[1]
                    
            print([ round(dists[k, treematch[k]], 4) for k in range(mybigK)])
            
            for k in range(mybigK):
                if (dists[k, treematch[k]] > .75):
                    freq[bigK] = np.array([0] * n)
                    treematch[k] = bigK
                    bigK = bigK+1
                    
            for k in range(mybigK):                    
                freq[treematch[k]] = freq[treematch[k]] + tmp_freq[k]            
        
    allfreqs = np.array([0] * n)
    for k in range(bigK):
        allfreqs = allfreqs + freq[k]
        freq[k] = freq[k]/sum(freq[k])        
    return((allfreqs, allK, tree2root, alpha0, mypi))




"""
WARNING: resets edge attribute "tree" and node attribute "pa". Number
        of trees may change

REQUIRE: graf has edge attribute "tree"

Generates forest F from P(F | mypi, graf) where F has K component trees 
with K fixed by nodewise Gibbs sampling.

INPUT: graf, igraph object
       mypi,  n--dim list of ordering
       tree2root,  K-dim 
       
       
OUTPUT: rootset,  K-dim list of new roots for trees

"""
def nodewiseSampleDP(graf, mypi, tree2root, alpha, beta, alpha0):
    n = len(graf.vs)
    m = len(graf.es)
    n2 = n*(n-1)/2
    
    ## DEBUG
    getTreeSizes(graf, tree2root)
    
    
    root_dict = {}
    for v in tree2root:
        root_dict[v] = 1
        
    
    mypi_inv = [0] * n
    for i in range(n):
        mypi_inv[mypi[i]] = i    
                
        
    """
    for v in tree2root:
        countSubtreeSizes(graf, v)
    
    all_tree_degs = [0] * n
    for i in range(n):
        mypa = graf.vs[i]["pa"]
        if (mypa != None):
            all_tree_degs[mypa] = all_tree_degs[mypa] + 1
            all_tree_degs[i] = all_tree_degs[i] + 1
    """
    all_tree_degs = getAllTreeDeg(graf)        
    assert sum(all_tree_degs) == 2*(n-len(tree2root))
    
    
    edge_ls = []
    curK = len(tree2root)
    for i in range(n-1):
       
        k = i + 1
        u = mypi[k]
        mypa = graf.vs[u]["pa"]
        uisroot = (mypa == None)
        
        nbs = graf.neighbors(u)
        nbs = [w for w in nbs if mypi_inv[w] < k]

        tree_degs = np.array([all_tree_degs[w] for w in nbs])
        root_adj = np.array([w in root_dict for w in nbs])
        pa_adj = np.array([w == mypa for w in nbs])
        
        tmp_p = beta*tree_degs + 2*beta*root_adj - beta*pa_adj + alpha
        
        new_root_wt = alpha0 * (m-n+curK+1-uisroot)/(n2-n+curK+1-uisroot) * \
            (beta*all_tree_degs[u] + beta*uisroot + alpha)/(beta+alpha)
        
        tmp_p = np.append(tmp_p, new_root_wt)
        
        """ draw a new parent for u"""
        nbs.append(-1)
        myw = choices(nbs, weights=tmp_p)[0]
        
        if (myw == -1):
            myw = None
        
        if (myw == mypa):
            if (mypa != None):
                edge_ls.append((u, mypa))
            
            continue
        
        """ modifying pa, all_tree_degs """
        if (myw != None):
            all_tree_degs[myw] = all_tree_degs[myw] + 1
            if (not uisroot):
                all_tree_degs[mypa] = all_tree_degs[mypa] - 1
            else:
                all_tree_degs[u] = all_tree_degs[u] + 1
                root_dict.pop(u)
                curK = curK - 1
            
            edge_ls.append((u, myw))
        else:
            ## u was not a root, became a root
            assert mypa != None
            root_dict[u] = 1
            curK = curK + 1
            all_tree_degs[u] = all_tree_degs[u] - 1
            all_tree_degs[mypa] = all_tree_degs[mypa] - 1


    assert len(edge_ls) == (n - curK)
    graf.es["tree"] = 0
    graf.vs["pa"] = None
    
    graf.es[graf.get_eids(edge_ls)]["tree"] = 1
    
    rootset = list(root_dict.keys())
    return(rootset)




"""
WARNING: resets edge attribute "tree" and node attribute "pa"

REQUIRE: graf has edge attribute "tree"

Generates forest F from P(F | mypi, graf) where F has K component trees 
with K fixed by nodewise Gibbs sampling.

INPUT: graf, igraph object
       mypi,  n--dim list of ordering
       
       
OUTPUT: None

"""
def nodewiseSamplePA(graf, mypi, alpha, beta, K):
    
    n  = len(graf.vs)
    
    mypi_inv = [0] * n
    for i in range(n):
        mypi_inv[mypi[i]] = i

    for k in range(K):
        countSubtreeSizes(graf, mypi[k])

    all_tree_degs = [0] * n
    for i in range(n):
        mypa = graf.vs[i]["pa"]
        if (mypa != None):
            all_tree_degs[mypa] = all_tree_degs[mypa] + 1
            all_tree_degs[i] = all_tree_degs[i] + 1
        
    edge_ls = []
    
    for i in range(n-K):
        k = K+i
        v = mypi[k]
        
        mypa = graf.vs[v]["pa"]
        assert mypa is not None
        ## adjust parent degree
        all_tree_degs[mypa] = all_tree_degs[mypa] - 1
        
        nbs = graf.neighbors(v)
        nbs = [w for w in nbs if mypi_inv[w] < k]

        tree_degs = [all_tree_degs[w] for w in nbs]
        tree_degs = np.array(tree_degs)

        root_adj = np.array([w in mypi[0:K] for w in nbs])
        
        if (K == 1):
            root_adj = 0
        
        """ generate new parent for u"""
        tmp_p = beta*tree_degs + 2*beta*root_adj + alpha
        myw = choices(nbs, weights=tmp_p)[0]

        edge_ls.append((v, myw))
        ## myw may potentially be mypa
        all_tree_degs[myw] = all_tree_degs[myw] + 1
        
    assert len(edge_ls) == (n - K)
    graf.es["tree"] = 0
    graf.vs["pa"] = None
    
    graf.es[graf.get_eids(edge_ls)]["tree"] = 1
    
    


"""
REQUIRE: graf.vs is in the order 0,1,2,3, ..., n-1
        graf.vs has "pa" attribute
        graf.es has "tree" attribute

INPUT: graf, igraph object 

NOTE: modifies "pa" attribute on nodes
      modifies "subtree_size" attributes on edges
"""
def sampleOrdering(graf, tree2root, alpha, beta, DP=False):
    
    K = len(tree2root)
    n = len(graf.vs)
    
    time3 = time.time()
    
    degs = getAllTreeDeg(graf)
    
    mypi = [0] * n
    
    tree_sizes = getTreeSizes(graf, tree2root)
    
    """ draw new roots for each subtree """
    for k in range(K):
        if (tree_sizes[k] == 1):
            graf.vs[tree2root[k]]["subtree_size"] = 1
            mypi[k] = tree2root[k]
            continue
        
        cur_root = tree2root[k]
        normalized_h = countAllHist(graf, cur_root)[0]
        
        deg_adj = (beta*degs + beta + alpha) * (beta*degs + alpha)
        if (K == 1):
            deg_adj = 1
        
        tmp_p = normalized_h*deg_adj
        mypi[k] = choices(range(n), tmp_p)[0]
        tree2root[k] = mypi[k]
        
        countSubtreeSizes(graf, root=mypi[k])
    
    if (DP):
        wts = [graf.vs[tree2root[k]]["subtree_size"] for k in range(K)]
        assert(sum(wts) == n)
        
        mypi[0] = tree2root[choices(range(K), wts)[0]]
        remain_nodes = [i for i in list(range(n)) if i != mypi[0]]
        assert mypi[0] not in remain_nodes
        
        mypi[1:n] = np.random.permutation(remain_nodes)
    else:
        remain_nodes = [i for i in list(range(n)) if i not in mypi[0:K]]
        mypi[K:n] = np.random.permutation(remain_nodes)
    
    mypi_inv = [0] * n
    for i in range(n):
        mypi_inv[mypi[i]] = i
    
    
    
    marked = {}
    
    if (DP):
        marked[mypi[0]] = 1
    else:
        for k in range(K):
            marked[mypi[k]] = 1
    
    for i in range(n-1):
        
        if (DP):
            k = 1 + i
        else:   
            k = K + i
            if (k >= n):
                break
            
        v = mypi[k]
        
        if (not DP): 
            assert v not in tree2root
            assert graf.vs[v]["pa"] != None
        
        if (v not in marked):
            ancs = getAncestors(graf, v)
            unmarked_ancs = [w for w in ancs if w not in marked]
            
            v_anc = unmarked_ancs[-1]            

            old_pos = mypi_inv[v_anc]
            mypi[old_pos] = v
            mypi[k] = v_anc
            mypi_inv[v_anc] = k
            mypi_inv[v] = old_pos
            
            marked[v_anc] = 1         
    
    if (DP):
        return((mypi, tree2root))    
    else:   
        return(mypi) 









"""    
INPUT: "vec1" is the possibly longer vector
        "vec2" contains elements of vec1 in a different order
        "pos_dict" a dictionary Elem(vec1) -> [n] giving positions
OUTPUT: a vector which contains the same elements as vec1
        the subvector that correspond to elements of vec2 is re-ordered
        according to vec2

WARNING: modifies "vec1" and "pos_dict" in place 
"""
def reorderSubvector(vec1, vec2, pos_dict):
    n = len(vec1)
    m = len(vec2)
    
    all_pos = [0] * m
    for i in range(m):
        all_pos[i] = pos_dict[vec2[i]]
    
    all_pos.sort()
        
    for i in range(m):
        vec1[all_pos[i]] = vec2[i]
        pos_dict[vec2[i]] = all_pos[i]
    
    return(vec1)














"""
DEPRECATED CODE below!


===========================

"""    



"""
DEPRECATED!

WARNING: resets edge attribute "tree" and node attribute "pa"

Generates forest F from P(F | mypi, graf) where F has K component trees 
with K fixed.

"""
def constrainedSamplePA(graf, mypi, alpha, beta, K):
    
    n = len(graf.vs)
    
    mypi_inv = [0] * n
    for i in range(n):
        mypi_inv[mypi[i]] = i
    
    """
    ## OG weight
    old_score = 0
    old_tree_degs = [0] * n
    for i in range(n-K):
        k = K+i
        v = mypi[k]
        
        nbs = graf.neighbors(v)
        nbs = [w for w in nbs if mypi_inv[w] < k]

        tree_degs = [old_tree_degs[w] for w in nbs]
        tree_degs = np.array(tree_degs)
        
        root_adj = [w in mypi[0:K] for w in nbs]
        root_adj = np.array(root_adj)        
            
        if (K == 1):
            root_adj = 0
        
        tmp_p = beta*tree_degs + 2*beta*root_adj + alpha
        
        if (i > 1):
            old_score = old_score + np.log(sum(tmp_p))
            if (sum(tmp_p) == 0):
                print((i, sum(old_tree_degs)))
            
        tree_edges = [e for e in graf.es[graf.incident(v)] if e["tree"] ]
        endpts = [otherNode(e, v) for e in tree_edges]
        bothls = [u for u in endpts if mypi_inv[u] < k]
        assert len(bothls) == 1
        
        old_tree_degs[bothls[0]] = old_tree_degs[bothls[0]] + 1
        old_tree_degs[v] = old_tree_degs[v] + 1
    ##
    """
    
    
    
    all_tree_degs = [0] * n
    edge_ls = []
    
    new_score = 0
    
    for i in range(n-K):
        k = K+i

        v = mypi[k]
        
        nbs = graf.neighbors(v)
        nbs = [w for w in nbs if mypi_inv[w] < k]

        tree_degs = [all_tree_degs[w] for w in nbs]
        ##tree_degs = [ sum( graf.es[graf.incident(w)]["tree"] ) for w in nbs]
        tree_degs = np.array(tree_degs)
        
        root_adj = [w in mypi[0:K] for w in nbs]
        root_adj = np.array(root_adj)
        
        if (K == 1):
            root_adj = 0
        
        tmp_p = beta*tree_degs + 2*beta*root_adj + alpha
        
        if (i > 1):
            new_score = new_score + np.log(sum(tmp_p))
        
        myw = choices(nbs, weights=tmp_p)[0]
        ##myw = choices(nbs)[0]
        
        ## add (v, myw) to the tree
        edge_ls.append((v, myw))
        all_tree_degs[myw] = all_tree_degs[myw] + 1
        all_tree_degs[v] = all_tree_degs[v] + 1
        ##graf.es[ graf.get_eids( [(v, myw)] ) ]["tree"] = 1
    
    assert len(edge_ls) == (n - K)
    
    """
    ratio = np.exp(new_score - old_score)
    print(ratio)
    
    u = random()
    if (u > ratio):
        constrainedSamplePA(graf, mypi, alpha, beta, K)
    """
    
    graf.es["tree"] = 0
    graf.vs["pa"] = None
    
    graf.es[graf.get_eids(edge_ls)]["tree"] = 1

    

"""
DEPRECATED

WARNING: resets edge attribute "tree" and node attribute "pa"

Generates forest F from P(F | mypi, graf) where F has Dirichlet Process
prior on the number of components


"""
def constrainedSampleDP(graf, mypi, alpha, beta, alpha0, MAXK=50):
    n = len(graf.vs)
    m = len(graf.es)
    n2 = n*(n-1)/2
    
    MAXK = min(MAXK, n)
    
    graf.es["tree"] = 0
    graf.vs["pa"] = None
    
    rootset = [mypi[0]]
    
    mypi_inv = [0] * n
    for i in range(n):
        mypi_inv[mypi[i]] = i    
        
    all_tree_degs = [0] * n
    edge_ls = []
    for i in range(n-1):
        k = i + 1
        v = mypi[k]
        
        nbs = graf.neighbors(v)
        nbs = [w for w in nbs if mypi_inv[w] < k]

        tree_degs = [all_tree_degs[w] for w in nbs]
        ##tree_degs = [ sum( graf.es[graf.incident(w)]["tree"] ) for w in nbs]
        tree_degs = np.array(tree_degs)
        
        root_adj = [w in rootset for w in nbs]
        root_adj = np.array(root_adj)
        
        tmp_p = beta*tree_degs + 2*beta*root_adj + alpha
        tmp_p = np.append(tmp_p, alpha0 * (m-n + MAXK)/(n2-n + MAXK) )
        
        myi = choices(range(len(nbs)+1), weights=tmp_p)[0]
        if (myi < len(nbs)):
            myw = nbs[myi]
            edge_ls.append((v, myw))
            ##graf.es[ graf.get_eids( [(v, myw)] ) ]["tree"] = 1
            all_tree_degs[myw] = all_tree_degs[myw] + 1
            all_tree_degs[v] = all_tree_degs[v] + 1
        else:
            print("")
            print("adding root")
            print((k, v, len(nbs)))
            print(tmp_p)
            print("")
            rootset.append(v)
        
        
        """
        ## DEBUG!!!   
        ## DEBUG!!!
        # for root in rootset:
        #     cur_tree = treeDFS(graf, root, range(n))
        #     if len([x for x in cur_tree if x in rootset]) > 1:
        #         print("BUG!")
        #         print((k, v, myi, myw))
        #         print(nbs)
        #         print(rootset)
        #         raise Exception("No!!")
        """
    assert len(edge_ls) == (n - len(rootset) )
        
    graf.es[graf.get_eids(edge_ls)]["tree"] = 1
    
    ## rejection sampling
    K = len(rootset)
    if (K > MAXK):
        return(constrainedSampleDP(graf, mypi, alpha, beta, alpha0, MAXK=2*MAXK))
    
    rej_ratio = 1
    for k in range(K-1):
        rej_ratio = rej_ratio * (m-n+k+2)/(m-n + MAXK) * (n2-n + MAXK)/(n2-n+k+2)

    U = np.random.uniform()
    
    ##print((K, rej_ratio))
    
    if (U < rej_ratio):
        return(rootset)
    else:
        return(constrainedSampleDP(graf, mypi, alpha, beta, alpha0, MAXK=MAXK))

