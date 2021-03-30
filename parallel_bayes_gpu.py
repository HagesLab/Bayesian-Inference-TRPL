#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:33:18 2020

@author: tladd
"""
## Define constants
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

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
        pN  *= ref                         # Update mutipliers
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


def marginalP(N, P, refs):                                   # Marginal P's
    pN   = np.prod(refs, axis=0)
    ind  = indexGrid(N, refs)
    marP = []
    for m in range(len(refs[0])):                            # Loop over axes
        Pm   = np.zeros(pN[m])
        for n in np.unique(ind[:,m]):                        # Loop over coord
            Pm[n] = P[np.where(ind[:,m] == n)].sum()         # Marginal P
        marP.append(Pm)                                      # Add to list
    return np.array(marP)

def plotMarginalP(marP, pN, minX, maxX, param_names):                     # Plot marginal P
    import matplotlib.pyplot as plt
    plt.clf()
    for m in np.where(pN > 1)[0]:                            # Loop over axes
        plt.figure(m, dpi=600)
        im = np.array([*range(pN[m])]) + 0.5                 # Coords
        X = minX[m] + (maxX[m]-minX[m])*im/pN[m]             # Param values
        plt.plot(X, marP[m], label = param_names[m])
        plt.xlim(minX[m], maxX[m])
        plt.ylim(0,1)
        plt.xlabel(param_names[m])
        plt.ylabel("Prob.")
        plt.legend(loc='best')
        
def export_marginal_P(marP, pN, minX, maxX, param_names):

    for m in np.where(pN > 1)[0]:
        print("Writing marginal {} file".format(param_names[m]))
        im = np.array([*range(pN[m])]) + 0.5                 # Coords
        X = minX[m] + (maxX[m]-minX[m])*im/pN[m]             # Param values
        np.savetxt(r"/home/cfai2304/super_bayes/" + experimental_data_filename + "_BAYRES_" + param_names[m] + ".csv", np.vstack((X, marP[m])).T, delimiter=",")

    return        

def bayes(model, N, P, refs, minX, maxX, init_params, sim_params, minP, data):        # Driver function
    global num_SMs
    global has_GPU
    for nref in range(len(refs)):                            # Loop refinements
        while not any(P > minP[nref]):
            print("Warning: prob. grid too diffuse for minP {}".format(minP[nref]))
            minP[nref] /= 2
            print("Retrying with minP {}".format(minP[nref]))
        N   = N[np.where(P > minP[nref])]                    # P < minP
        N   = refineGrid(N, refs[nref])                      # Refine grid
        Np  = np.prod(refs[nref])                            # Params per set
        lnP = np.zeros(len(N))                               # Likelihoods
        print("ref level, N: ", nref, len(N))
        
        # TODO: Determine block size from GPU info instead of refinement?
        # Np cannot be modified! indexGrid assumes a certain value of Np
        for n in range(0, len(N), Np):                       # Loop over blks
            Nn  = N[n:n+Np]                                  # Cells block
            ind = indexGrid(Nn,  refs[0:nref+1])             # Get coordinates
            X   = paramGrid(ind, refs[0:nref+1], minX, maxX) # Get params
            
            if has_GPU:
                plI = model(X, sim_params, init_params, sim_params[2], num_SMs)[-1]
            else:
                plI = model(X, sim_params, init_params)[1][-1]

            Pbk = np.zeros(len(X)) # P's for block
            times, values, std = data
            
            # TODO: Match experimental data timesteps to model timesteps
            values = values[::sim_params[4]]
            std = std[::sim_params[4]]
            for i in range(len(plI[0])):
                F = plI[:,i]
                sig  = modelErr(F, refs[nref])
                sg2  = 2*(sig.max()**2 + std[i]**2)
                Pbk -= (F-values[i])**2 / sg2 + np.log(np.pi*sg2)/2
   
            lnP[n:n+Np] = Pbk
            
        # TODO: Better normalization scheme
        print(lnP)
        P = np.exp(lnP + 1000*np.log(2) - np.log(len(lnP)) - np.max(lnP))
        P  /= np.sum(P)                                      # Normalize P's
    return N, P


#-----------------------------------------------------------------------------#
import csv
from numba import cuda

def get_data(exp_file, scale_f=1):
    with open(exp_file, newline='') as file:
        ifstream = csv.reader(file)
        t = []
        PL = []
        uncertainty = []
        for row in ifstream:
            t.append(float(row[0]))
            PL.append(float(row[1]))
            uncertainty.append(float(row[2]))

    t = np.array(t)
    
    PL = np.array(PL) * scale_f
    uncertainty = np.array(uncertainty) * scale_f
    return (t, PL, uncertainty)

if __name__ == "__main__":
    # simPar
    Time    = 50                             # Final time (ns)
    Length  = 1500                            # Length (nm)
    lambda0 = 704.3                           # q^2/(eps0*k_B T=25C) [nm]
    L   = 2 ** 7                                # Spatial points
    T   = 2000                                # Time points
    plT = 10                                  # Set PL interval (dt)
    pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
    tol = 5                                   # Convergence tolerance
    MAX = 500                                  # Max iterations
    pT = tuple(np.array(pT)*T//100)
    simPar = (Length, Time, L, T, plT, pT, tol, MAX)
    
    # iniPar
    a  = 1e17/(1e7)**3                        # Amplitude
    l  = 100                                  # Length scale [nm]
    iniPar = (a, l)
    
    # This code follows a strict order of parameters:
    # matPar = [N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda]
    param_names = ["n0", "p0", "mu_n", "mu_p", "B", "Sf", "Sb", "tau_n", "tau_p", "rel. permitivity^-1"]
    unit_conversions = np.array([(1e7)**-3,(1e7)**-3,(1e7)**2/(1e9)*.02569257,(1e7)**2/(1e9)*.02569257,(1e7)**3/(1e9),(1e7)/(1e9),(1e7)/(1e9),1,1,lambda0])
    
    ref1 = np.array([1,8,1,1,8,8,1,8,8,1])
    ref2 = np.array([1,4,1,1,4,4,1,4,4,1])
    ref3 = np.array([1,4,1,1,4,4,1,4,4,1])
    refs = np.array([ref1, ref2, ref3])                         # Refinements
    
    minX = np.array([1e8, 2e14, 10, 10, 1e-12, 1e2, 1e-6, 10, 10, 13.6**-1])                        # Smallest param values
    maxX = np.array([1e8, 1e17, 10, 10, 1e-9, 5e4, 1e-6, 100, 100, 13.6**-1])                        # Largest param values
    
    #minP = np.array([0, 0.01, 0.01])                 # Threshold P
    minP = np.array([0] + [0.01 for i in range(len(refs) - 1)])

    N    = np.array([0])                              # Initial N
    P    = np.array([1.0])                            # Initial P

    experimental_data_filename = "less_noisy pvSim example.csv"
    
    # Pre-checks
    from sys import exit
    try:
        num_params = len(param_names)
        assert (len(unit_conversions) == num_params), "Unit conversion array is missing entries"
        assert (len(minX) == num_params), "Missing min param values"
        assert (len(maxX) == num_params), "Missing max param values"  
        assert all(minX > 0), "Invalid param values"
        assert all(minX <= maxX), "Min params larger than max params"
        # TODO: Additional checks involving refs
            
        print("Starting simulations with the following parameters:")
        for i in range(num_params):
            if minX[i] == maxX[i]:
                print("{}: {}".format(param_names[i], minX[i]))
                
            else:
                print("{}: {} to {}".format(param_names[i], minX[i], maxX[i]))
        
        print("Refinement levels:")
        for i in range(num_params):
            print("{}: {}".format(param_names[i], refs[:,i]))        
        e_data = get_data(experimental_data_filename, scale_f=1e-37) # [carr/cm^2 s] to [carr/nm^2 ns]
        print("\nExperimental data - {}".format(experimental_data_filename))
        print(e_data)

        try:
            has_GPU = cuda.detect()
        except Exception:
            has_GPU = False

        if has_GPU: 
            device = cuda.get_current_device()
            num_SMs = getattr(device, "MULTIPROCESSOR_COUNT")
            from pvSimPCR import pvSim
        else:
            print("No GPU detected - reverting to CPU simulation")
            num_SMs = -1
            from pvSim import pvSim

        #_continue = input("Continue? (y/n)")
        
        #if not (_continue == 'y'): raise KeyboardInterrupt("Aborted")
        
    except Exception as oops:
        print(oops)
        exit(0)
        
    minX *= unit_conversions
    maxX *= unit_conversions

    import time
    clock0 = time.time()
    N, P = bayes(pvSim, N, P, refs, minX, maxX, iniPar, simPar, minP, e_data)
    print("Bayesim took {} s".format(time.time() - clock0))
    marP = marginalP(N, P, refs)
    #plotMarginalP(marP, np.prod(refs,axis=0), minX * (unit_conversions ** -1), maxX * (unit_conversions ** -1), param_names)
    export_marginal_P(marP, np.prod(refs,axis=0), minX * (unit_conversions ** -1), maxX * (unit_conversions ** -1), param_names)





