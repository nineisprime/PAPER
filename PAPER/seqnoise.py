#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 20:02:15 2022

@author: minx
"""


from PAPER.tree_tools import *
import PAPER.gibbsSampling as gibbsSampling

from igraph import *
import numpy as np
from random import choices
import math

def gibbsFullSeq(graf, 
                 Burn=40, M=50, 
                 gap=1, 
                 K=1,
                 initpi=None, 
                 seq=True, timed=False, display=True, **params):
        
    st = time.time()
    n = len(graf.vs)
    m = len(graf.es)
    
    alpha = params["alpha"]
    beta = params["beta"]
    
    if (initpi is None):
        wilsonTree(graf)
        v = choices(range(n))[0]
    
        countSubtreeSizes(graf, v)
        tree2root = [v]
        
        mypi = gibbsSampling.sampleOrdering(graf, tree2root, alpha, beta)
    else:
        mypi = initpi
    
    mypi_inv = {}
    for i in range(n):
        mypi_inv[mypi[i]] = i
    
    freq = [0] * n

    for i in range(Burn + M):     

        st1 = time.time()        
        #print(i)
        
        # Step 2 of Gibbs: Sampling the tree
        nodewiseSampleSeq(graf, pi=mypi, pi_inv=mypi_inv, 
                         K=K, **params)
        
        tree2root = mypi[0:K]
        #print("MH")

        # Step 1 of Gibbs: Metropolis hastings by sampling non-root
        start = time.time() 
        MHOrdering(graf, pi=mypi, pi_inv=mypi_inv, M=n,
                   display=timed, **params)
        end = time.time()


        if ( i == Burn+M - 1 and timed): 
            print("** Total time MHOrdering at i = {}: {:.4f} **".format(i,end-start))
        
        # Step 3 of Gibbs: sample root
        sampleRootSeq(graf, pi=mypi, pi_inv=mypi_inv, freq=freq,  
                      k0=15, M0=10, **params)
        
        if (np.random.rand() > 0.99 and timed):
            print("mcmc iter: %d  time: %.4f" % (i, time.time() - st1))
            
       
    return((freq, mypi))





def nodewiseSampleSeq(graf, pi, pi_inv, K, 
                     beta=1, alpha=0,
                     theta=1, 
                     tbeta=1, talpha=0, 
                     eta=0,
                     debug_flag = False,
                     **params):
    """
    Generates new forest for a given ordering by sampling
    a new parent for each node. Used in fixed K setting.

    Require: graf.es has "tree" attribute

    Parameters
    ----------
    graf : igraph object
        Input graph; "tree" edge attribute and "pa" node
        attributes are modified in place.
    pi : list
        Given ordering of the nodes.
    alpha : float
        APA Parameter.
    beta : float
        APA Parameter.
    eta: float
        Tree-edge deletion parameter.
    K : int
        Num of clusters.

    Returns
    -------
    None.

    """
    
    
    n  = len(graf.vs)

    for k in range(K):
        countSubtreeSizes(graf, pi[k])

    all_tree_degs = getAllTreeDeg(graf)
        
    edge_ls = []
    
    ## DEBUG
    max_termc = 0
    
    for i in range(n-K):
        
        """ v = pi[k]  is current node to be assigned a parent"""
        k = K+i
        v = pi[k]
        mypa = graf.vs[v]["pa"]
        assert mypa is not None
        
        ## adjust parent degree
        all_tree_degs[mypa] = all_tree_degs[mypa] - 1
        nbs = [w for w in graf.neighbors(v) if pi_inv[w] < k]

        """ generate new parent for u"""        
        if eta == 0:
                
            
            root_adj = np.array([w in pi[0:K] for w in nbs])

            if (K == 1):
                root_adj = 0


            if len(nbs) == 1:
                myw = mypa
            else:
                tmp_logp = [0] * len(nbs)
                
                noise_logp = [0] * len(nbs)
                
                for ii in range(len(nbs)):
                    w = nbs[ii]
                    logp = calcElderProb(graf, w, k, n-1, pi, pi_inv,
                                    theta, talpha, tbeta, adopt=1) - \
                            calcElderProb(graf, w, k, n-1, pi, pi_inv,
                                      theta, talpha, tbeta, adopt=2)
                            
                    noise_logp[ii] = logp
                            
                    logp = logp + np.log( beta*all_tree_degs[w] + \
                                     2*beta*root_adj + alpha )
                    
                        
                    tmp_logp[ii] = logp                    
                
                tmp_logp = np.array(tmp_logp)
                tmp_logp = tmp_logp - np.mean(tmp_logp)
                tmp_p = np.exp(tmp_logp)
                                                    
                myw = choices(nbs, weights=tmp_p)[0]
            
            if (myw != mypa):
                old_edge = graf.get_eids( [(v, mypa)] )
                graf.es[old_edge]["tree"] = 0
                graf.es[ graf.get_eids( [(v, myw)] )]["tree"] = 1
                
            #edge_ls.append((v, myw))
                
        else:
            """ BEGIN setting with deletion """
            prev_nodes = [w for w in pi[0:k]]
            
            
            old_edge = graf.get_eids( [(v, mypa)] )[0]
            old_pa_fake = graf.es[old_edge]["del"]
            
            if (old_pa_fake):
                nbs.remove(mypa)            
            
            if len(prev_nodes)==1:
                myw = mypa
                edge_ls.append((v, myw))
            else:
            
                root_adj = 0 ## Change for multiple roots
    
                if (K == 1):
                    root_adj = 0
            
                tmp_logp = [-float("inf")] * len(prev_nodes)
                
                FILTER_CONST = 3
                filter_prob = min(FILTER_CONST/np.sqrt(k), 1)
                filter_flag = (np.random.rand() > filter_prob) and (len(nbs) > 0)
                #filter_flag = 0
                
                for ii in range(len(prev_nodes)):
                    w = prev_nodes[ii]
                    
                    
                    if (filter_flag and (w not in nbs and w != mypa)):
                        continue
                    
                    
                    if (w in nbs):
                        adopt_flag = 2
                    else:
                        adopt_flag = 3
                    
                    
                    logp = calcElderProb(graf, w, k, n-1, pi, pi_inv,
                                        theta, tbeta, talpha, adopt=1) - \
                            calcElderProb(graf, w, k, n-1, pi, pi_inv,
                                          theta, tbeta, talpha, adopt=adopt_flag)
                    
                    logp = logp + np.log( beta*all_tree_degs[w] + \
                                         2*beta*root_adj + alpha )
                           
                        
                    if (w in nbs):
                        logp = logp + np.log(1-eta)
                    else:
                        logp = logp + np.log(eta)
                        
                    tmp_logp[ii] = logp
    

                    
                tmp_logp = np.array(tmp_logp)
                tmp_logp = tmp_logp - np.max(tmp_logp)
                tmp_p = np.exp(tmp_logp)
                tmp_p = tmp_p/sum(tmp_p)

                term_a = sum([tmp_p[jj] for jj in range(len(prev_nodes)) \
                                  if prev_nodes[jj] in nbs or prev_nodes[jj] == mypa])
                term_b = sum(tmp_p) - term_a


                if (filter_flag):
                    #print((filter_prob, term_a))
                    
                    assert filter_prob < term_a
                    
                    filter_adjust_a = (1 - filter_prob/term_a) / (1 - filter_prob)
                    filter_adjust_b = 1/(1 - filter_prob)
                    
                    assert filter_adjust_a > 0
                    
                    for ii in range(len(prev_nodes)):
                        w = prev_nodes[ii]
                        if (w not in nbs and w != mypa):
                            tmp_p[ii] = tmp_p[ii] * filter_adjust_a
                        else:
                            tmp_p[ii] = tmp_p[ii] * filter_adjust_b
                        
                        #print(tmp_p[ii])
                        #assert tmp_p[ii] < 1


                """ DEBUG"""
                term_c = np.sqrt(k)*(term_b/(term_a+term_b))
                
                if (term_c > max_termc and len(nbs) > 0):
                    max_termc = term_c
                
                if (term_c > FILTER_CONST and len(nbs) > 0):
                    print((k, term_c))
                """end"""
                
                    
                myw = choices(prev_nodes, weights=tmp_p)[0]
                
                if (myw != mypa):
                    if (old_pa_fake):
                        graf.delete_edges(graf.es[old_edge])
                    else:
                        graf.es[old_edge]["tree"] = 0
                    
                    if (myw not in nbs):
                        
                        graf.add_edges( [(v, myw)] )
                        graf.es[-1]["del"] = 1
                        graf.es[-1]["tree"] = 1
                    else:
                        graf.es[graf.get_eids( [(v, myw)] )]["tree"] = 1
                                        
            """END setting with deletion"""
        
        ## myw may potentially be mypa
        
        all_tree_degs[myw] = all_tree_degs[myw] + 1
        graf.vs[v]["pa"] = myw
        
        
    
    if (np.random.rand() > .995 and eta > 0):
        print("term c: %.3f" % max_termc)
        
        
    assert sum(graf.es["tree"]) == (n-K)




# CHANGES BELOW: generalized noise function
# dependency of theta through acceptance probability only  
def MHOrdering(graf, pi, pi_inv, M=50, gap=1, 
               display=False, **params):
    """
    Condition on the forest, generate a new global ordering
    by swapping a pair of non-root nodes at each step with 
    the Metropolis Hastings algorithm.
    
    Require: graf.vs has "pa" attribute; graf.es has "tree" attribute

    """    
    
    theta = params["theta"]
    talpha = params["talpha"]
    tbeta = params["tbeta"]
    
    n = len(graf.vs)

    for i in range(M):
        
        # Check if new ordering is consistent with treehist
        cons = False
        while(cons == False):
            pair2swap = np.random.choice(list(range(1,n)), size=2, replace=False)  
            cons = nonRootSwapConsistent(graf, pi, pi_inv, pair2swap)
          
        #a = min(calcPairSwapAcceptProb(graf, pi, pi_inv, pair2swap, theta),1) # need generalized theta version
        a = min(calcSwapAcceptProb(graf, pi, pi_inv, pair2swap, 
                                   theta, talpha, tbeta), 1)
        
        if (np.random.uniform() < a):
            # swap: new pi
            pi_inv[pi[pair2swap[0]]], pi_inv[pi[pair2swap[1]]] = pi_inv[pi[pair2swap[1]]], pi_inv[pi[pair2swap[0]]]
            pi[pair2swap[0]], pi[pair2swap[1]] = pi[pair2swap[1]], pi[pair2swap[0]] 




# dependency of theta through calcRootSeqLogProb only
def sampleRootSeq(graf, pi, pi_inv, freq, k0=15, M0=10, **params):
    """
    Potentially updates the first k0 elements of pi
    
    Updates pi_inv correspondingly
    
    """
    theta = params["theta"]
    tbeta = params["tbeta"]
    talpha = params["talpha"]
    
    k0 = min(len(pi), k0)
    
    subpi = [0] * k0

    old_subpi = pi[0:k0]
    sub_hist = countSubtreeHist(graf, old_subpi, pi[0])
    
    prev_logp = calcRootSeqLogProb(graf, pi, pi_inv, 
                                   pi, pi_inv, k0, 
                                   theta, talpha, tbeta)
    
    for i in range(M0):
        
        ## uniformly sample a root with prob prop to sub_hist
        subpi[0] = choices(old_subpi, sub_hist)[0]
        
        if (subpi[0] != pi[0]):
            adjustSubtreeSizes(graf, pi[0:k0], root=subpi[0])
        
        remain_nodes = [u for u in pi[0:k0] if u != subpi[0]]
        subpi[1:k0] = np.random.permutation(remain_nodes)

        ## adjust subpi in place
        subpi_inv = {}
        for j in range(k0):
            subpi_inv[subpi[j]] = j
            
        marked = {}
        marked[subpi[0]] = 1

        for j in range(1, k0):
            v = subpi[j]
            
            if (v not in marked):
                v_anc = getUnmarkedAncestor(graf, v, marked)
                
                #assert v_anc != v
                
                subpi[j], subpi[ subpi_inv[v_anc] ] = v_anc, v
                subpi_inv[ v ], subpi_inv[v_anc] = subpi_inv[v_anc], j
                
            marked[subpi[j]] = 1
            
            
        ## calculate acceptance prob
        cur_logp = calcRootSeqLogProb(graf, subpi, subpi_inv, 
                                      pi, pi_inv, k0, 
                                      theta, talpha, tbeta)
        
        
        
        a = min(1, np.exp(cur_logp - prev_logp))
        #print(a)

        if (np.random.uniform() < a):
            pi[0:k0] = subpi
            for j in range(k0):
                pi_inv[pi[j]] = j
            prev_logp = cur_logp
        elif (subpi[0] != pi[0]):
            ## revert "pa" attribute
            adjustSubtreeSizes(graf, pi[0:k0], root=pi[0])
            
        
        freq[pi[0]] = freq[pi[0]] + 1
        
        







def calcSwapAcceptProb(graf, pi, pi_inv, pair2swap, theta, talpha, tbeta):
    
    ## DEBUG
    #print("")
    #print("new swap!")
    #print(pair2swap)
    
    j = min(pair2swap)
    k = max(pair2swap)
    
    pj = pi[j]
    pk = pi[k]

    pa_pj = graf.vs[pj]["pa"]
    pa_pk = graf.vs[pk]["pa"]
    
    before_term1 = calcElderProb(graf, pa_pj, j, k, pi, pi_inv, 
                                theta, talpha, tbeta) + \
        calcElderProb(graf, pa_pk, j, k, pi, pi_inv, theta, talpha, tbeta)
    #before_term1 = 0
    before_term = before_term1 + \
        calcPairProb(graf, pj, pk, j, k, pi, pi_inv, theta, talpha, tbeta)
        
        
        
    after_term1 = calcElderProb(graf, pa_pj, j, k, pi, pi_inv, 
                                theta, talpha, tbeta, swap=True) + \
        calcElderProb(graf, pa_pk, j, k, pi, pi_inv, 
                      theta, talpha, tbeta, swap=True)
    #after_term1 = 0
    after_term = after_term1 + \
        calcPairProb(graf, pk, pj, j, k, pi, pi_inv, theta, talpha, tbeta,
                     swap=True)

    #if (before_term1 != after_term1):
    #    print((before_term1, after_term1))
    #assert(abs(before_term1 - after_term1) < 0.000001)

    return(np.exp(after_term - before_term))

"""
Either pj = pi[start] and pk = pi[end] or
       pj = pi[end] and pk = pi[start]
       

"""
def calcPairProb(graf, u, v, start, end, pi, pi_inv, 
                 theta, talpha, tbeta, swap=False):
    
    if swap:
        pi_inv[pi[start]], pi_inv[pi[end]] = \
            pi_inv[pi[end]], pi_inv[pi[start]]
        pi[start], pi[end] = pi[end], pi[start]
        
    
    assert (u == pi[start] and v == pi[end]) 
    
    pa_u = graf.vs[u]["pa"]
    pa_v = graf.vs[v]["pa"]
    
    nb_edges_u = graf.es[graf.incident(u)]
    noise_edges_u = [e for e in nb_edges_u if not e["tree"]]
    noise_nbs_u = [otherNode(e, u) for e in noise_edges_u]
    noise_nbs_u_pre = [w for w in noise_nbs_u if pi_inv[w] < start]
    noise_nbs_u_post = [w for w in noise_nbs_u if pi_inv[w] > start  \
                        and pi_inv[w] < end]
    
    logprod = 0
    
    ## compute first term of u
    
    if (start > 1):
        for w in noise_nbs_u_pre:
        

            Dw = pastDegree(graf, w, start, pi, pi_inv) if tbeta > 0 else 0

            
            logprod = logprod +  \
                np.log(theta*(tbeta*Dw + talpha)) - \
                np.log((2*tbeta+talpha)*start - \
                       (2*tbeta + theta*tbeta*Dw + theta*talpha))
                    
        
        Dpa = pastDegree(graf, pa_u, start, pi, pi_inv) if tbeta > 0 else 0

        logprod = logprod + \
            np.log( 2*(start-1)*tbeta + start*talpha) - \
            np.log( (2*tbeta+talpha)*start - \
                       (2*tbeta + theta*tbeta*Dpa + theta*talpha) )
            
    ## compute second term of u
    
    for w in noise_nbs_u_post:
        logprod = logprod + \
            np.log( theta*(tbeta + talpha)) - \
            np.log((2*tbeta+talpha)*pi_inv[w] - (2*tbeta + theta*tbeta + theta*talpha))
    
    ## compute term of v
    
    nb_edges_v = graf.es[graf.incident(v)]
    noise_edges_v = [e for e in nb_edges_v if not e["tree"]]
    
    tmp = [otherNode(e, v) for e in noise_edges_v]
    noise_nbs_v = [w for w in tmp if pi_inv[w] <= end]

    for w in noise_nbs_v:
        
        Dw = pastDegree(graf, w, end, pi, pi_inv) if tbeta > 0 else 0
        logprod = logprod + np.log( theta*(tbeta*Dw + talpha) ) - \
            np.log( (2*tbeta+talpha)*end - (2*tbeta + theta*tbeta*Dw + theta*talpha))
    
    Dpa = pastDegree(graf, pa_v, end, pi, pi_inv) if tbeta > 0 else 0
    logprod = logprod + \
        np.log( 2*(end-1)*tbeta + end*talpha) - \
        np.log((2*tbeta+talpha)*end - \
                   (2*tbeta + theta*tbeta*Dpa + theta*talpha))
    

    if swap:
        pi_inv[pi[start]], pi_inv[pi[end]] = \
            pi_inv[pi[end]], pi_inv[pi[start]]
        pi[start], pi[end] = pi[end], pi[start]
        assert pi_inv[pi[start]] == start

    return(logprod)


"""
Calculates contribution to likelihood from
u, [start, end], with respect to pi

We discard terms not specific to u

adopt = 0   no change between (u, pi[start])
adopt = 1   force (u, pi[start]) to be tree edge, even if
                (u, pi[start]) is not an edge in the graph
adopt = 2   force (u, pi[start]) to be noise edge
"""
def calcElderProb(graf, u, start, end, pi, pi_inv,
                  theta, talpha, tbeta, swap=False, adopt=0):
    
    assert pi_inv[u] < start
        
    if swap:
        pi_inv[pi[start]], pi_inv[pi[end]] = \
            pi_inv[pi[end]], pi_inv[pi[start]]
        pi[start], pi[end] = pi[end], pi[start]
        
        assert pi_inv[pi[start]] == start
    
    
    flag = 0
    if (start == 1):
        assert u == pi[0]
        assert adopt == 0
        flag = 1
        start = 2
    
    ## staging the calculation
    
    nb_edges = graf.es[graf.incident(u)]
    tree_edges = [e for e in nb_edges if e["tree"] ]
    noise_edges = [e for e in nb_edges if not e["tree"] ]
    
    tree_nbs = [otherNode(e, u) for e in tree_edges]
    tree_nbs_int = [v for v in tree_nbs if pi_inv[v] > start and pi_inv[v] <= end]
    
    if (adopt==0 and (pi[start] in tree_nbs)):
        tree_nbs_int.insert(0, pi[start])
    if (adopt==1):
        tree_nbs_int.insert(0, pi[start])
    
    noise_nbs = [otherNode(e, u) for e in noise_edges]
    noise_nbs_int = [v for v in noise_nbs if pi_inv[v] > start and pi_inv[v] <= end]
    
    if (adopt==0 and (pi[start] in noise_nbs)):
        noise_nbs_int.insert(0, pi[start])
    if (adopt==2):
        assert (pi[start] in tree_nbs) or (pi[start] in noise_nbs)
        noise_nbs_int.insert(0, pi[start])
    
    noise_times = [pi_inv[v] for v in noise_nbs_int]
    noise_times.sort()
    
    if (adopt==3):
        # the edge (pi[start], u) cannot be noise
        assert pi[start] not in noise_nbs
        
    
    
    if (pi[start] in tree_nbs_int):
        t_ls = []
    else:   
        t_ls = [start]
    t_ls.extend([pi_inv[v] for v in tree_nbs_int])
    
    t_ls.append(end+1)
    t_ls.sort()
    
    d0 = pastDegree(graf, u, start, pi, pi_inv) if tbeta > 0 else 0
    
    ## calculate noise likelihood term
    noise_iter = 0
    
    logprob = 0

    dl = d0


    for l in range(len(t_ls)-1):
        
        
        if (l > 0):
            assert pi[t_ls[l]] in tree_nbs_int
        
        dl = d0 + l
        
        while (noise_iter < len(noise_times) and 
               noise_times[noise_iter] < t_ls[l+1]):
            
            my_t = noise_times[noise_iter]
            
            #if (my_t == start):
            #    noise_iter = noise_iter + 1
            #    continue
            assert my_t >= 2
            
            num = theta*(tbeta*dl + talpha)
            denom = (2*tbeta+talpha)*my_t - \
                (2*tbeta + theta*tbeta*dl + theta*talpha)
                
            logprob = logprob + np.log(num/denom)
            
            noise_iter = noise_iter + 1
            
            
        Cterm = (2*tbeta + theta*tbeta*dl + theta*talpha)/  \
            (2*tbeta + talpha)    
            
        #print((Cterm, dl, theta, t_ls[l]))
        #print((d0, dl, start))
            
        gamma_term = (t_ls[l+1] - t_ls[l])*np.log(2*tbeta + talpha)
        gamma_term = gamma_term + math.lgamma(t_ls[l+1] - Cterm) - \
            math.lgamma(t_ls[l] - Cterm)
            
        logprob = logprob + gamma_term
        
            
        ## second term adjusting for tree
        if (pi[t_ls[l]] in tree_nbs_int):
            logprob = logprob + np.log(2*tbeta*(t_ls[l]-1) + t_ls[l]*talpha) - \
                np.log(2*tbeta*(t_ls[l]-1) + t_ls[l]*talpha - theta*(tbeta*dl +talpha) )
            
        
    if (flag):
        start = 1
    
    if swap:
        pi_inv[pi[start]], pi_inv[pi[end]] = \
            pi_inv[pi[end]], pi_inv[pi[start]]
        pi[start], pi[end] = pi[end], pi[start]
        assert pi_inv[pi[start]] == start
    
    return(logprob)



        
        
def calcRootSeqLogProb(graf, subpi, subpi_inv, pi, pi_inv, k0,
                       theta, talpha, tbeta):
    
    ## tree_degs[j] = tree degree of node subpi[j]
    tree_degs = [1] * k0
    
    logp = 0
    
    #print((talpha, tbeta))
    
    for t in range(2,k0):
        
        #nb_edges = graf.es[graf.incident(subpi[t])]
        nb_edges = [graf.es[e] for e in graf.incident(subpi[t]) ] 
        #tree_edges = [e for e in nb_edges if e["tree"] ]
        noise_edges = [e for e in nb_edges if not e["tree"] ]
    
        #tree_nbs = [otherNode(e, subpi[t]) for e in tree_edges]
        #tree_nbs = [v for v in tree_nbs if v in subpi_inv and subpi_inv[v] < t]
    
        tree_nbs = [graf.vs[subpi[t]]["pa"]]
    
        noise_nbs = [otherNode(e, subpi[t]) for e in noise_edges]
        noise_nbs = [v for v in noise_nbs if v in subpi_inv and subpi_inv[v] < t]
        
        for j in range(t):
            myp = 0
            if (subpi[j] in noise_nbs):
                myp = np.log( theta*(tbeta*tree_degs[j] + talpha) ) 
            if ((subpi[j] not in noise_nbs) and (subpi[j] not in tree_nbs)):
                myp = np.log( (2*tbeta+talpha)*t - \
                             (2*tbeta + theta*tbeta*tree_degs[j]+ theta*talpha))
            logp = logp + myp
            
            
        #assert len(tree_nbs) == 1
        
        tree_degs[subpi_inv[tree_nbs[0]]] = tree_degs[subpi_inv[tree_nbs[0]]] + 1
    
    return(logp)





