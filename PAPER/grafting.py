#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:44:30 2021

@author: minx
"""

from tree_tools import *
import gibbsSampling

from igraph import *
import numpy as np
from random import *
from estimateAlpha import *



def gibbsGraftToConv(graf, DP=False, K=1, alpha=0, beta=0, alpha0=5, 
                     Burn=30, M=50, gap=1, MAXITER=100, tol=0.1, size_thresh=0.01):
    
    n = len(graf.vs)
    m = len(graf.es)
    graf2 = graf.copy()
    
    if (alpha == 0 and beta == 0):
        beta = 1
        alpha = estimateAlphaEM(graf, display=False)
     
    print("estimated alpha: {0}".format(alpha))
    
    options = {"Burn": Burn, "M": M, "gap": gap, "alpha": alpha, 
               "beta": beta, "display": False, "size_thresh": size_thresh}
               
    
    if (not DP):
        res = gibbsGraft(graf, K=K, initroots=None, **options)
        
        res1 = gibbsGraft(graf2, K=K, initroots=None, **options)
    else:   
        res = gibbsGraftDP(graf, alpha0=alpha0, initroots=None, **options)
        
        res1 = gibbsGraftDP(graf2, alpha0=alpha0, initroots=None, **options)

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
            res = gibbsGraft(graf, K=K, initroots=res[-1], **options)
            
            res1 = gibbsGraft(graf2, K=K, initroots=res1[-1], **options)
        else:
            res = gibbsGraftDP(graf, alpha0=res[-2],  
                              initroots=res[-1], **options)
            
            res1 = gibbsGraftDP(graf2, alpha0=res[-2],
                               initroots=res1[-1], **options)        
    
    allfreq = allfreq + allfreq1
    allfreq = allfreq/sum(allfreq)
    
    return((allfreq, res, res1))



"""
INPUT: 

OUTPUT: 0: allfreqs, sum of all freqs normalized
        1. freq, if K==1, then a single vector
                if K>1, then a dictionary mapping k to vectors
        2: n
        3: tree2root, set of roots, K--dim vector
"""
def gibbsGraft(graf, Burn=40, M=50, gap=1, alpha=0, beta=1, K=1,
               display=True, size_thresh=0.01, initroots=None):
    
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
                    if (dists[k, treematch[k]] > 0.75):
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
    return((allfreqs, freq, node_tree_coo, tree2root))



"""
INPUT: 

OUTPUT: 0: allfreqs, sum of all freqs normalized
        1: freqs, dictionary of clusters
        2: allK, posterior distribution of number of component trees
        3: alpha0
        4: tree2root, set of roots, K--dim vector
"""
def gibbsGraftDP(graf, Burn=30, M=50, gap=1, alpha=0, beta=1, alpha0=5, 
                 display=True, size_thresh=0.01, initroots=None):
    
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
            
            
        sizes = getTreeSizes(graf, tree2root)
        sizes_sorted = -np.sort( - np.array(sizes))
        sizes_args = np.argsort( - np.array(sizes))
        
        K = len(tree2root)

        tree2root_sorted = [0] * len(tree2root)
        for k in range(K):
            tree2root_sorted[k] = tree2root[sizes_args[k]]
        
        
        alpha0tilde = drawAlpha0tilde(K, n, alpha0/(alpha+2*beta))
        alpha0 = alpha0tilde*(alpha+2*beta)    
    
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
            #print("{0}, {1}".format(mybigK, bigK))
            
            for k in range(mybigK):
                for kk in range(bigK):
                    if (sum(freq[kk] > 0)):
                        
                        distr1 = np.array(freq[kk]/sum(freq[kk]) )                        
                        distr2 = np.array(tmp_freq[k])

                        dists[k, kk] = sum(np.abs(distr1 - distr2))/2
                    else:
                        dists[k, kk] = 0
                    
            treematch = scipy.optimize.linear_sum_assignment(dists)[1]
                    
            if (display):
                print([ round(dists[k, treematch[k]], 4) for k in range(mybigK)])
            
            for k in range(mybigK):
                if (dists[k, treematch[k]] > 0.75):
                    freq[bigK] = np.array([0] * n)
                    treematch[k] = bigK
                    bigK = bigK+1
                    
            for k in range(mybigK):                    
                freq[treematch[k]] = freq[treematch[k]] + tmp_freq[k]
                    
    allfreqs = np.array([0] * n)        
    for k in range(bigK):
        allfreqs = allfreqs + freq[k]
        freq[k] = freq[k]/sum(freq[k])
    
    allfreqs = allfreqs/sum(allfreqs)
    return((allfreqs, freq, allK, alpha0, tree2root))


    
"""
INPUT: 

OUTPUT: [0]  integer, index of new root
"""
def sampleRoot(graf, v, alpha, beta, single_root=False, degs=None):
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
    



"""
INPUT: 

OUTPUT: [0]  integer, node index of new parent of u
        [1]  new tree2root list of roots
"""

def graftSubtreeDP(graf, u, tree2root, alpha=0, beta=1, alpha0=1, degs=None):
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
        
    
    
"""



"""
    
def graftSubtreePA(graf, u, tree2root, alpha=0, beta=1, degs=None):
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
        
    
    
    
    
    
    
    
    
    
"""
DEPRECATED CODE below!


===========================

"""    
    
    


    

"""
DEPRECATED!

TODO: incorporate multiple trees
OUTPUT: tuple (root distribution,  number of iterations)
"""
def mcmcInferToConvergence(graf, Burn=500, M_rw=50, M_graft=50, init=False, 
              alpha=1, beta=0, gap=99, tol=0.05, display=False):
    
    n = len(graf.vs)
    
    M = 2*gap
    
    graf2 = graf.copy()
    
    chain1 = mcmcInfer(graf, M=3*M, Burn=Burn, M_rw=M_rw, 
                       M_graft=M_graft, alpha=alpha, beta=beta, gap=gap, display=display)
    chain2 = mcmcInfer(graf2, M=3*M, Burn=Burn, M_rw=M_rw, 
                       M_graft=M_graft, alpha=alpha, beta=beta, gap=gap, display=display)
    
    freq1 = chain1
    freq2 = chain2
    
    hellin = (1/2)*np.sum( (freq1**(1/2) - freq2**(1/2))**2 )
    
    ii = 0
    MAX_ITER = 800
    while (hellin > tol and ii < MAX_ITER):
        if (ii % 50 == 0):
            print([ii, hellin])
        ii = ii + 1
        chain1 = chain1 + mcmcInfer(graf, M=M, Burn=5, M_rw=M_rw, 
                       M_graft=M_graft, alpha=alpha, beta=beta, init=True, gap=gap)
        chain2 = chain2 + mcmcInfer(graf2, M=M, Burn=5, M_rw=M_rw, 
                       M_graft=M_graft, alpha=alpha, beta=beta, init=True, gap=gap)
    
        freq1 = chain1/sum(chain1)
        freq2 = chain2/sum(chain2)
        
        hellin = (1/2)*np.sum( (freq1**(1/2) - freq2**(1/2))**2 )
        
    all_freq = freq1 + freq2
    all_freq = all_freq/sum(all_freq)
    
    return((all_freq, ii))



"""
DEPRECATED

Implements the root-and-graft walk

INPUT: "graf" is an igraph object
        "init" is True if edges in "graf" contains "tree" attribute
              e.g. sum( graf.es["tree"] ) should equal n - 1
        "M" total number of regular Gibbs steps
          "gap" number of steps to skip before updating inference output


OUTPUT: "freq", if K=1 n-dimensional vector of probabilities,
                  if K > 1, then a dictionary [K] -> n-dimen vector of prob
"""
def mcmcInfer(graf, M=4000, Burn=500, M_rw=50, M_graft=50, init=False, 
              alpha=1, beta=0, gap=100, K=1, display=False):
    
    n = len(graf.vs)
    m = len(graf.es)

    if (not init):    
        wilsonTree(graf)
        v = choices(range(n))[0]
        #bfsTree(graf, v)
        
        countSubtreeSizes(graf, v)  ## sets "pa" attribute
        #print("Wilson found.")
        
        
        tree2root = [0] * K
        root2tree = {}        
        
        tree2root[0] = v
        root2tree[v] = 0
        
        
        ## if K > 1, cut the wilson Tree into K pieces
        ## Iteratively select a random node "w" as a new root
        
        for i in range(K-1):
            w = choices(range(n))[0]

            while (graf.vs[w]["pa"] == None):
                w = choices(range(n))[0]
            
            w_root = w
            
            while (graf.vs[w_root]["pa"] != None) :
                w_root = graf.vs[w_root]["pa"]
                
            w_pa = graf.vs[w]["pa"]
            
            ## cut the edge between w and w_pa
            ## update "pa" attribute for w
            ## update tree_size, tree2root, root2tree
            ## recompute subtree_size for w_root

            my_e = graf.get_eids([(w, w_pa)])[0]
            
            graf.es[my_e]["tree"] = False
            graf.vs[w]["pa"]= None
        
            
            tree2root[i+1] = w
            root2tree[w] = i+1
            
            countSubtreeSizes(graf, w_root)    
            
        
    else:
        ## If initialized: what to do
        ## TO DO: add 
        if (K > 1):
            raise Exception("Cannot use initialization when K > 1")

    
    print(root2tree)
    print(tree2root)
    
    graft_wts = graf.degree(graf.vs)
    
    co_occur = np.zeros((n, n))
    node_tree_coo = np.zeros((n, K))
    
    if (K == 1):
        freq = [0] * n
    else:
        freq = {}
        for k in range(K):
            freq[k] = [0] * n
    
    freqcounter = 0
        
    
    for i in range(Burn + M):            
        
        if (display):
            print(i)
        
        tmp = random()
        if (tmp > 0.99995):
            cur_time = time.time()
        
        for k in range(M_rw):
            
            
            for kk in range(K):
                oldroot = tree2root[kk]
                tree2root[kk] = subtreeRW(graf, tree2root[kk])
                
                root2tree.pop(oldroot)
                root2tree[tree2root[kk]] = kk
                        
        for k in range(M_graft):
             
            eu = randrange(0, m)
            xunif = randrange(2)
            if (xunif == 0):
                u = graf.es[eu].source
            else:
                u = graf.es[eu].target

            
            if (u in tree2root):
                continue
            
            
            graftSubtree(graf, u, tree2root, alpha=alpha, beta=beta)
        
        
        ## update inference information
        ## "freq", "node_tree_coo"
        if (i >= Burn and (i - Burn) % gap == 0):
            
            if (K == 1):
                freq = freq + countAllHist(graf, tree2root[0])[0]
            else:   
                
                tree_match = list(range(K))
                
                if (freqcounter == 0):
                    
                    # first time recording results
                    for k in range(K):
                        freq[k] = countAllHist(graf, tree2root[k])[0]
                        
                else:
                    
                    # match each tree in the forest to existing records
                    taken = {}
                    for k in range(K):
                        tmp_freq = countAllHist(graf, tree2root[k])[0]
                        
                        #print(tmp_freq)
                        
                        dists = [0] * K
                        for kk in range(K):
                            if kk in taken:
                                dists[kk] = 10
                            else:
                                dists[kk] = sum(np.abs(np.array(tmp_freq) - np.array(freq[kk]/sum(freq[kk])) ))
                        kstar = np.argmin(dists)
                        taken[kstar] = 1
                        
                        tree_match[k] = kstar
                        
                        freq[kstar] = freq[kstar] + tmp_freq
                
                
                freqcounter = freqcounter + 1
                
                
                ## compute co-occurrence between node and tree
                for ii in range(n):
                    ants = getAncestors(graf, ii)
                    myroot = ants[-1]
                    my_K = root2tree[myroot]
                    
                    my_Kstar = tree_match[my_K]
                    
                    node_tree_coo[ii, my_Kstar] = node_tree_coo[ii, my_Kstar] + 1    
                
        
        if (tmp > 0.99995):
            print(time.time() - cur_time)

    if (K==1):
        freq = freq/sum(freq)
    else:
        for k in range(K):
            freq[k] = freq[k]/sum(freq[k])
        
    return((freq, node_tree_coo, co_occur, freqcounter))
        


    
"""
DEPRECATED

u is assumed to NOT be root        
  we have that graf.vs[root]["pa"] = None       
    
INPUT: u, root are node indices
OUTPUT: index of the parent of u (could be same as before)

modifies graf: changes u["pa"] and the "subtree_size" of old and new ancestors
      changes the "tree" value of two edges
"""
def graftSubtree(graf, u, tree2root, alpha=1, beta=0):
    
    #assert u != root
    #assert graf.vs[root]["pa"] is None
    assert not (u in tree2root)
    
    u_size = graf.vs[u]["subtree_size"]
    
    ## get all potential parents of u
    u_pas = []
    u_wts = []
    u_edge_ixs = graf.incident(u)
    
    old_pa = graf.vs[u]["pa"]
    
    assert graf.vs[old_pa]["subtree_size"] > u_size
    
    for eid in u_edge_ixs:
        my_e = graf.es[eid]
        
        utilde = otherNode(my_e, u) ## utilde is a potential parent
        
        #if (graf.vs[utilde]["pa"] == None):
        #    my_wt = 1/u_size
        #else :
        ants = getAncestors(graf, utilde, u)
        if (ants == -1):
            continue
            
        #utilde_root = ants[-1]
        #utilde_tree_size = tree_sizes[root2tree[utilde_root]]
            
        if (old_pa == utilde):
            ants_big_sizes = np.array(graf.vs[ants]["subtree_size"])
            ants_sm_sizes = ants_big_sizes - u_size  
            deg_adj = treeDegree(graf, utilde)-1
            
        else:
            ants_sm_sizes = np.array(graf.vs[ants]["subtree_size"])
            ants_big_sizes = ants_sm_sizes + u_size
            deg_adj = treeDegree(graf, utilde)
           
            
        cur_tsize = ants_sm_sizes[-1]
        
        if (cur_tsize == 1):
            size_adj = 0
        else:   
            size_adj = np.sum(np.log(2*beta*( np.array(
                range(cur_tsize-1, cur_tsize + u_size-1)) ) + alpha))
        
        #print(size_adj)
        #print((size_adj, u_size))
        my_wt = (alpha + beta*deg_adj ) * \
            np.exp(np.sum(np.log(ants_sm_sizes) - np.log(ants_big_sizes) - size_adj))
                   
                
        
        u_pas.append(utilde) 
        u_wts.append(my_wt)
        
    new_pa = choices(u_pas, weights=u_wts)[0]
    
    if (new_pa == old_pa):
        return(old_pa)
    else:
        graf.vs[u]["pa"] = new_pa
        old_edge_ix = graf.get_eids( [(u, old_pa)] )[0]
        graf.es[old_edge_ix]["tree"] = False
    
        new_edge_ix = graf.get_eids( [(u, new_pa)] )[0]
        graf.es[new_edge_ix]["tree"] = True
    
        #if (graf.vs[new_pa]["pa"] != None):
        ants = getAncestors(graf, new_pa, u)
        for ix in ants:
            graf.vs[ix]["subtree_size"] = graf.vs[ix]["subtree_size"] + u_size
                
        #if (graf.vs[old_pa]["pa"] != None):
        old_ants = getAncestors(graf, old_pa, u)
        for ix in old_ants:
            graf.vs[ix]["subtree_size"] = graf.vs[ix]["subtree_size"] - u_size
            assert graf.vs[ix]["subtree_size"] > 0
            
        return(new_pa)




"""
DEPRECATED

Performs one step of random root walk

REQUIRE: graf has node attribute "subtree_size"

INPUT: "graf" is igraph object, "v" is an integer in range(n) 
                                  represents root
OUTPUT: integer representing the new node
"""
def subtreeRW(graf, v):
    edge_ixs = graf.incident(v)
    n = graf.vs[v]["subtree_size"]
    
    nbs = []
    for eid in edge_ixs:
        my_e = graf.es[eid]
        if (not my_e["tree"]):
            continue
        
        nbs.append(otherNode(my_e, v))
    
    
    if (len(nbs) == 0):
        return(v)
    
    next_v = choices(nbs, weights=graf.vs[nbs]["subtree_size"])[0]
    
    graf.vs[v]["subtree_size"] = n - graf.vs[next_v]["subtree_size"]
    graf.vs[next_v]["subtree_size"] = n
    
    graf.vs[v]["pa"] = next_v
    graf.vs[next_v]["pa"] = None
    
    return(next_v)