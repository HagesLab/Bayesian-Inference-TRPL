#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:33:18 2020

@author: tladd
"""

import numpy as np

def indexGrid(N, refs):                        # Arrays of cell coordinates
    cN  = N.copy()                             # Copy of cell indexes
    K   = len(refs)                            # Num params
    M   = len(refs[0])
    pN  = np.ones(M, int)
    indexes = np.zeros((len(N),M), int)        # Initialize indexes
    for k in range(K):                         # Loop over refinement levels
        ref = refs[K-k-1]                      # Loop backwards
        ind = []                               # Indexes for current level
        for m in range(len(ref)):              # Loop over directions
            ind.append(cN%ref[m])              # Index for this direction
            cN //= ref[m]                      # Divide out fom N
        indexes += np.array(ind).T*pN          # Accumulate indexes from level
        pN  *= refs[k]                         # Update mutipliers
    return indexes

def paramGrid(ind, refs, minX, maxX):          # Arrays of parameters
    pN  = np.prod(refs, axis=0)                # Scale for indexes
    X   = minX + (maxX-minX)*(ind + 0.5)/pN    # Get params
    return X

def refineGrid (N, ref):                       # Refine grid
    siz = np.prod(ref)                         # Size of refined block
    reN = np.arange(siz)                       # Refined indexes
    N   = np.add.outer(reN, N*siz)             # 2D array of indexes
    return N.flatten(order='F')                # Return flattened array

def modelErr(F, ref):
    N   = np.prod(ref)
    pN = 1
    err = []
    for m in range(len(ref)):
        dF  = np.abs(F - np.roll(F, -pN))      # Absolute differences
        dk  = ref[m]*pN                        # Step size
        for n in range(pN):                    # Need pN passes
            dF[dk-pN+n:N:dk] = 0               # Zero wrapped entries
        err.append(dF.max())
        pN *= ref[m]                           # Update counter
    return np.array(err)

def forwardSim(model, X, ref, data):
    times, values, std = data                  # Unpack data tuple
    lnP = np.zeros(len(X))                     # Sum of log(likelihood)
    for n in range(len(times)):
        F    = model(times[n], X)
        sig  = modelErr(F, ref)
        sg2  = 2*(sig.max()**2 + std[n]**2)
        lnP -= (F-values[n])**2 / sg2 + np.log(np.pi*sg2)/2
    return lnP

def marginalP(N, P, refs):                                   # Marginal P's
    pN   = np.prod(refs, axis=0)
    ind  = indexGrid(N, refs)
    marP = []
    for m in range(len(refs[0])):                            # Loop over axes
        Pm   = np.zeros(pN[m])
        for n in np.unique(ind[:,m]):                               # Loop over coord
            Pm[n] = P[np.where(ind[:,m] == n)].sum()         # Marginal P
        marP.append(Pm)                                      # Add to list
    return np.array(marP)

def plotMarginalP(marP, pN, minX, maxX):                     # Plot marginal P
    import matplotlib.pyplot as plt
    plt.clf()
    for m in range(len(pN)):                                 # Loop over axes
        im = np.array([*range(pN[m])]) + 0.5                 # Coords
        X = minX[m] + (maxX[m]-minX[m])*im/pN[m]             # Param values
        plt.plot(X, marP[m], label = str(m))
    plt.xlim(np.min(minX), np.max(maxX))
    plt.ylim(0,1)
    plt.legend(loc='best')

def bayes(model, N, P, refs, minX, maxX, minP, data):        # Driver function
    for nref in range(len(refs)):                            # Loop refinements
        N   = N[np.where(P > minP[nref])]                    # P < minP
        N   = refineGrid(N, refs[nref])                      # Refine grid
        Np  = np.prod(refs[nref])                            # Params per set
        lnP = np.zeros(len(N))                               # Likelihoods
        print("ref level, N: ", nref, len(N))
        for n in range(0, len(N), Np):                       # Loop over blks
            Nn  = N[n:n+Np]                                  # Cells block
            ind = indexGrid(Nn,  refs[0:nref+1])             # Get coordinates
            X   = paramGrid(ind, refs[0:nref+1], minX, maxX) # Get params
            Pbk = forwardSim(model, X, refs[nref], data)     # P's for block
            lnP[n:n+Np] = Pbk
        P = np.exp(lnP + 680 - np.max(lnP))
        P  /= np.sum(P)                                      # Normalize P's
    return N, P


#-----------------------------------------------------------------------------#


def testData(t, parms):
    f = np.zeros(len(t))
    e = np.ones(len(t))*0.01                          # Uncertainty
    for p in parms[1:]:
        f += np.exp(-p*t)                             # Function
    f *= parms[0]
    return (t, f, e)                                  # Data tuple
def testModel(t, parms):
    f = np.zeros(len(parms))
    for p in parms.T[1:]:
        f += np.exp(-p*t)                             # Function
    f *= parms[:,0]
    return f

if __name__ == "__main__":
    ref1 = np.array([3,3,1])
    ref2 = np.array([3,3,1])
    ref3 = np.array([3,3,1])
    refs = [ref1, ref2, ref3]                         # Refinements
    minX = np.array([0, 0, 10])                        # Smallest param values
    maxX = np.array([1, 1, 10])                        # Largest param values
    minP = np.array([0, 0.01, 0.01])                  # Threshold P
    N    = np.array([0])                              # Initial N
    P    = np.array([1.0])                            # Initial P
    mP   = ()                                         # Forward model params
    t = np.linspace(0, 5, 100+1)                      # Event tmies
    data = testData(t, np.array([0.7, 0.5, 10]))        # Test data
    N, P = bayes(testModel, N, P, refs, minX, maxX, minP, data)
    marP = marginalP(N, P, refs)
    plotMarginalP(marP, np.prod(refs,axis=0), minX, maxX)




