#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 13:45:51 2021

@author: minx
"""


from PAPER.tree_tools import *
from igraph import *
import numpy as np
from random import choices

import time


      
def estimateAlphaEM(graf, K=1, maxiter=40, alpha_init=1, display=True):
    """
    Assumes beta=1, estimates alpha via approximate EM algorithm. Assumes
    PAPER model for input graph. Warning: very slow when maximum degree is
    over 10,000. 

    Parameters
    ----------
    graf : igraph object
        Input graph.
    K : int, optional
        Number of clusters. The default is 1.
    maxiter : int, optional
        Maximum number of EM iterations. The default is 40.
    alpha_init : float, optional
        Initial value of alpha. The default is 1.
    display : boolean, optional
        The default is True.

    Returns
    -------
    float: estimated alpha.

    """
    m = len(graf.es)
    n = len(graf.vs)
    
    degseq = graf.degree()
    max_deg = max(degseq)
    
    thetahat = (m - (n-K)) / ( n*(n-1)/2 - (n-K) )
    
    #alpha_init = 1
    alphap = alpha_init

    for i in range(maxiter):
        ## condprob[k, s] = P( D_T(v) = s | D_G(v) = k)
        condprob = np.zeros((max_deg+1, max_deg+1))
        ## cumprob[k, s] = P( D_T(v) > s | D_G(v) = k)
        cumprob = np.zeros((max_deg+1, max_deg+1))
        
        for k in range(1, max_deg+1):
            
            if (k > 5*n*thetahat):
                
                condprob[k, k] = 1
                for s in range(k-1, 0, -1):
                    ## Bin_{n-s, theta}(k-s) / Bin_{n-(s+1), theta}(k-(s+1)) 
                    ## PA_alpha(s) / PA_alpha(s+1)
                    condprob[k, s] = condprob[k, s+1] * (n-s)*thetahat \
                                        * (1/(k-s))  \
                                        * (s+3+2*alphap)/(s+alphap)                                         
            
            else:
                
                condprob[k, 1] = 1
                for s in range(2, k+1):
                    ## Bin_{n-s, theta}(k-s) / Bin_{n-(s-1), theta}(k-(s-1)) 
                    ## PA_alpha(s) / PA_alpha(s-1)
                    condprob[k, s] = condprob[k, s-1] / ( (n-(s-1))*thetahat) \
                                        * (k-(s-1))  \
                                        * ((s-1)+alphap)/((s-1)+3+2*alphap)
                               
            
            
            ## normalize
            condprob[k, ] = condprob[k, ]/sum(condprob[k, ])
     
            ## compute cumulative conditional
            cumprob[k, k] = 0
            for s in range(k-1, 0, -1):
                cumprob[k, s] = cumprob[k, s+1] + condprob[k, s+1]
            
            #for s in range(1, k):
            #    cumprob[k, s] = sum( condprob[k, (s+1):(k+1)] )
 
        
        Mterm = [0] * max_deg       
        for j in range(max_deg):
            for ii in range(n):   
                Mterm[j] = Mterm[j] + cumprob[degseq[ii], j+1]
        
        Mterm = np.array(Mterm)
        Mterm = Mterm * n / sum(Mterm)
        
        ##print(Mterm)
        
        ## update alpha
        
        
        alpha_next = optimizeAlpha(Mterm, n, alphap)
    
        if (display):   
            print("Optimizing...")
            print((i, alpha_next, thetahat))
    
        if (abs(alpha_next - alphap) < 1e-4):
            if (display):
                print("EM converged.")
            break
        
        alphap = alpha_next    
    
    return(alphap)


"""    


"""  
def optimizeAlpha(Mterm, n, alphainit=1):
    """
    Use radient descent with line search to compute

        max_alpha sum_j log(j + alpha) M[j] - sum_k log(2 (k-2) + (k-1)alpha )

    Parameters
    ----------
    Mterm : nparray
        M in optimization objective.
    n : int
        Num of nodes.
    alphainit : float, optional
        initial value. The default is 1.

    Returns
    -------
    alpha : float, argmax.

    """
    
    max_deg = len(Mterm)
    
    maxiter = 50
    
    alpha = alphainit
    ##alpha_next = alpha
    
    ls_a = 0.1
    ls_b = 0.5
    stepsize = 1
    
    ks = np.array( range(3, n+1) )
    js = np.array( range(1, max_deg+1))
    
    
    for i in range(maxiter):
        
        grad1 = sum(1/(js + alpha) * Mterm)
        grad1 = grad1 - sum( (ks - 1)/(2*(ks - 2) + (ks - 1)*alpha) )
        
        if (abs(grad1) < 1e-4):
            break
        if (alpha < 1e-4 and grad1 < 1e-4):
            break
        
        for ii in range(maxiter):
            
            alphap = max(alpha + stepsize * grad1, 0)
            diff = alphap - alpha
            
            
            old_val = sum(np.log(js + alpha) * Mterm)  \
                      - sum( np.log(2*(ks-2) + (ks-1)*alpha) )
            new_val = sum(np.log(js + alphap) * Mterm) \
                      - sum( np.log(2*(ks-2) + (ks-1)*alphap) )
            
            ##print((stepsize, grad1, new_val, old_val))
            
            if (new_val > old_val + ls_a * stepsize * grad1 * diff):
                alpha = alphap
                break
            else:
                stepsize = stepsize * ls_b
    
    
    return(alpha)



def drawAlpha0tilde(K, n, alpha0tilde, a=1, b=0.1, M=150):
    """
    Given K, draw alpha0 by gibbs Sampling.
    In APA_(alpha, beta, alpha0), define alpha0tilde = alpha0/(2 beta + alpha)
    We give alpha0tilde prior Gamma(a, b)    

    Parameters
    ----------
    K : int
        Num of clusters.
    n : int
        Num of nodes.
    alpha0tilde : float
        previous alpha0tilde value.
    a : float, optional
        prior parameter. The default is 1.
    b : float, optional
        prior parameter. The default is 0.1.
    M : int, optional
        gibbs iteration. The default is 150.

    Returns
    -------
    float; next alpha0tilde value.

    """
    
    cur_a0 = alpha0tilde
    a0_ls = []
    
    for i in range(M):
        x = np.random.beta(cur_a0+1, n)
        
        Z = choices( [0,1], weights=[(a+K-1)/(n*(b-np.log(x))), 1] )[0]
        
        if (Z == 0):
            cur_a0 = np.random.gamma(a+K, 1/(b-np.log(x)))
        else:
            cur_a0 = np.random.gamma(a+K-1, 1/(b-np.log(x)))
            
        a0_ls.append(cur_a0)
        #print((x, Z, cur_a0))

    return(np.mean(np.array(a0_ls)))




def optimizeAlpha0(K, n, maxiter=50):
    """
    Compute

    max_alpha0 (K-1)log(alpha0) - sum_{i=1}^{n-1} log( alpha0 + 2i )    

    Parameters
    ----------
    K : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    maxiter : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    None.

    """
    
    alpha0 = 1
    stepsize = .5
    ls_a = 0.2
    ls_b = 0.5
    
    js = np.array( range(1, n) )
    for i in range(maxiter):
        grad1 = (K-1)/alpha0 - sum(1/(alpha0 + 2*js))
        
        if (abs(grad1) < 1e-4):
            break
        if (alpha0 < 1e-4 and grad1 < 1e-4):
            break
        
        for ii in range(maxiter):
            alphap = max(alpha0 + stepsize*grad1, 1e-6)
            diff = alphap - alpha0
            
            
            old_val = (K-1)*np.log(alpha0) - sum( np.log(2*js + alpha0))
            new_val = (K-1)*np.log(alphap) - sum( np.log(2*js + alphap))
            
            if (new_val > old_val + ls_a*stepsize*grad1*diff):
                alpha0 = alphap
                break
            else:
                stepsize = stepsize * ls_b
        
    return(alpha0)
    
    