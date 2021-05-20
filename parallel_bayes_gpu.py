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
    global do_log
    pN  = np.prod(refs, axis=0)                # Scale for indexes
    X_lin   = minX + (maxX-minX)*(ind + 0.5)/pN    # Get params
    X_log = minX * (maxX/minX)**((ind + 0.5)/pN)
    return X_lin * (1-do_log) + X_log * do_log

def refineGrid (N, ref):                       # Refine grid
    siz = np.prod(ref)                         # Size of refined block
    reN = np.arange(siz)                       # Refined indexes
    N   = np.add.outer(reN, N*siz)             # 2D array of indexes
    return N.flatten(order='F')                # Return flattened array

def modelErr2(F, ref):
    N   = np.prod(ref)
    pN = 1
    err = np.zeros((len(ref), len(F[0])))
    for m in range(len(ref)):
        dF  = 1.5*np.abs(F - np.roll(F, -pN, axis=0))	   # Absolute differences
        dk  = ref[m]*pN                        # Step size
        for n in range(pN):                    # Need pN passes
            dF[dk-pN+n:N:dk] = 0               # Zero wrapped entries
        err[m] = np.amax(dF, axis=0)
        pN *= ref[m]                           # Update counter
    return err


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

def maxP(N,P,refs, minX, maxX):
    pN = np.prod(refs, axis=0)
    ind = indexGrid(N,refs)
    X = paramGrid(ind, refs, minX, maxX)
    print(X[np.argmax(P)])
    print("P = {}".format(P[np.argmax(P)]))

def cov_P(N,P,refs, minX, maxX):
    global param_names
    pN   = np.prod(refs, axis=0)
    ind  = indexGrid(N, refs)
    iterables = np.where(pN > 1)[0]
    
    for q in range(len(iterables)):
        for r in range(q):
            pID_1 = iterables[q]
            pID_2 = iterables[r]
            
            cov_P = np.zeros((pN[pID_1], pN[pID_2]))
            
            for i in np.unique(ind[:,pID_1]):
                for j in np.unique(ind[:,pID_2]):
                    # q = P[np.where(np.logical_and(ind[:,pID_1] == i, ind[:,pID_2] == j))]
                    cov_P[i,j] = P[np.where(np.logical_and(ind[:,pID_1] == i, ind[:,pID_2] == j))].sum()
            
            m = pID_1
            im = np.array([*range(pN[m])]) + 0.5                 # Coords
            X1   = minX[m] + (maxX[m]-minX[m])*(im)/pN[m]    # Get params
            X2   = minX[m] * (maxX[m]/minX[m])**(im/pN[m])
            X = X1 * (1 - do_log[m]) + X2 * do_log[m]
            cov_P = np.hstack((X.reshape((len(X),1)), cov_P))
            
            m = pID_2
            im = np.array([*range(pN[m])]) + 0.5                 # Coords
            X1   = minX[m] + (maxX[m]-minX[m])*(im)/pN[m]    # Get params
            X2   = minX[m] * (maxX[m]/minX[m])**(im/pN[m])
            X = np.append(np.array([-1]), (X1 * (1 - do_log[m]) + X2 * do_log[m]), axis=0)
            cov_P = np.vstack((X.reshape((1, len(X))), cov_P))
            
            print("Writing covariance file {} and {}".format(param_names[pID_1], param_names[pID_2]))
            np.save(wdir + out_filename +\
                    "_BAYRES_" + param_names[pID_1] + "-" + param_names[pID_2] + ".npy", cov_P)
    return
        
def export_marginal_P(marP, pN, minX, maxX, param_names):

    for m in np.where(pN > 1)[0]:
        print("Writing marginal {} file".format(param_names[m]))
        im = np.array([*range(pN[m])]) + 0.5                 # Coords
        #X = minX[m] + (maxX[m]-minX[m])*im/pN[m]             # Param values
        X_lin   = minX[m] + (maxX[m]-minX[m])*(im)/pN[m]    # Get params
        X_log = minX[m] * (maxX[m]/minX[m])**((im)/pN[m])
        X =  X_lin * (1-do_log[m]) + X_log * do_log[m]

        np.savetxt(wdir + out_filename + "_BAYRES_" + param_names[m] + ".csv", np.vstack((X, marP[m])).T, delimiter=",")

    return        

def find_neighbors(N, nref, refs):
    applied_refs = np.prod(refs[0:nref], axis=0) # not refs[0:nref+1], as #nref hasn't been applied yet
    search_range = 1 # The maximum distance neighbors of parameter i can be
    new_N = set(N) # set() can absorb duplicates at minimal cost
    
    for i in applied_refs:
        if i > 1:
            og_range = search_range # How far away neighbors of parameter i are expected to be
            search_range *= i # Size of neighborhood where parameter i+1 is constant.
            for n in N:
                # Exceeding this means parameter i was a boundary value i.e. has only one neighbor
                min_N = n - n % search_range
                max_N = min_N + search_range - 1
                
                new_n = n - og_range
                if new_n >= min_N: new_N.add(new_n)
                
                new_n = n + og_range
                if new_n <= max_N: new_N.add(new_n)
                
    new_N = list(new_N)
    return np.array(new_N, dtype=np.int32)

def bayes(model, N, P, refs, minX, maxX, init_params, sim_params, minP, data):        # Driver function
    global num_SMs
    global has_GPU
    global include_neighbors, init_mode
    global GPU_GROUP_SIZE
    for nref in range(len(refs)):                            # Loop refinements
        N   = N[np.where(P > minP[nref])]                    # P < minP

        if nref > 0 and include_neighbors:
            N = find_neighbors(N, nref, refs)

        N   = refineGrid(N, refs[nref])                      # Refine grid
        Np  = np.prod(refs[nref])                            # Params per set
        lnP = np.zeros(len(N))                               # Likelihoods
        print("ref level, N: ", nref, len(N))

        X = np.empty((len(N), len(refs[0])))
        
        # TODO: Determine block size from GPU info instead of refinement?
        # Np cannot be modified! indexGrid assumes a certain value of Np
        for n in range(0, len(N), Np):                       # Loop over blks
            Nn  = N[n:n+Np]                                  # Cells block
            ind = indexGrid(Nn,  refs[0:nref+1])             # Get coordinates
            #X   = paramGrid(ind, refs[0:nref+1], minX, maxX) # Get params
            X[n:n+Np] = paramGrid(ind, refs[0:nref+1], minX, maxX) # Get params


            ## OVERRIDE: MAKE SRH TAUS EQUAL
            #X[:,8]=X[:,7]
            #X[:,3]=X[:,2]

        #X[:,8] = X[:,7]
        #X[:,3] = X[:,2]

            #plI = np.empty((len(X), sim_params[3] // sim_params[4] + 1), dtype=np.float32)

        plI = np.empty((len(X), len(init_params) *(sim_params[3] // sim_params[4] + 1)), dtype=np.float32)
        
        for blk in range(0,len(X),GPU_GROUP_SIZE):
            
            if has_GPU:
                plI[blk:blk+GPU_GROUP_SIZE] = model(X[blk:blk+GPU_GROUP_SIZE], sim_params, init_params, TPB, num_SMs, init_mode=init_mode)[-1]
            else:
                plI[blk:blk+GPU_GROUP_SIZE] = model(X[blk:blk+GPU_GROUP_SIZE], sim_params, init_params)[1][-1]
        
        times, values, std = data
        #sys.exit()    
        if LOG_PL:
            plI[plI<bval] = bval
            plI = np.log(plI)
        # TODO: Match experimental data timesteps to model timesteps
        sig_sq = 1 / (len(plI) - len(X[0])) * np.sum((plI - values) ** 2, axis=0)
        Pbk = np.zeros(len(X)) # P's for block
        for n in range(0, len(N), Np):
            Pbk2 = Pbk[n:n+Np]
            plI2 = plI[n:n+Np]
            #sig = modelErr2(plI2, refs[nref])
            #sg2 = 2*(np.amax(sig, axis=0)**2 + std**2)
            #sg2 = 2 * (sig_sq + std ** 2)
            sg2 = 2 * sig_sq
            Pbk2 -= np.sum((plI2-values)**2 / sg2 + np.log(np.pi*sg2)/2, axis=1)
            lnP[n:n+Np] = Pbk2

        """
        UNC_GROUP_SIZE = int(1e8 / len(N))     
        for i in range(0, len(plI[0]), UNC_GROUP_SIZE):
            sig = modelErr2(plI[:,i:i+UNC_GROUP_SIZE], refs[nref])
            sg2 = 2*(np.amax(sig, axis=0)**2 + std[i:i+UNC_GROUP_SIZE]**2)
            Pbk -= np.sum((plI[:,i:i+UNC_GROUP_SIZE]-values[i:i+UNC_GROUP_SIZE])**2 / sg2 + np.log(np.pi*sg2)/2, axis=1)
            
            for i in range(len(plI[0])):
                F = plI[:,i]
                sig  = modelErr(F, refs[nref])
                sg2  = 2*(sig.max()**2 + std[i]**2)
                Pbk -= (F-values[i])**2 / sg2 + np.log(np.pi*sg2)/2
            
        #lnP[n:n+Np] = Pbk
        lnP = Pbk
        """    
            
        # TODO: Better normalization scheme
        P = np.exp(lnP + 1000*np.log(2) - np.log(len(lnP)) - np.max(lnP))
        P  /= np.sum(P)                                      # Normalize P's
    return N, P


#-----------------------------------------------------------------------------#
import csv
from numba import cuda
import sys
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
  
    if LOG_PL:
        PL += bval
        uncertainty /= PL
        PL = np.log(PL)
    return (t, PL, uncertainty)

def get_initpoints(init_file, scale_f=1e-21):
    with open(init_file, newline='') as file:
        ifstream = csv.reader(file)
        initpoints = []
        for row in ifstream:
            assert len(row) == L, "Error: length of initial condition does not match simPar: L\n IC:{}, L:{}".format(len(row), L)
            initpoints.append(row)

    return np.array(initpoints, dtype=float) * scale_f

if __name__ == "__main__":
    # simPar
    Time    = 250                             # Final time (ns)
    Length  = 2000                            # Length (nm)
    lambda0 = 704.3                           # q^2/(eps0*k_B T=25C) [nm]
    L   = 2 ** 7                                # Spatial points
    T   = 8000                                # Time points
    plT = 8                                  # Set PL interval (dt)
    pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
    tol = 5                                   # Convergence tolerance
    MAX = 500                                  # Max iterations
    pT = tuple(np.array(pT)*T//100)
    simPar = (Length, Time, L, T, plT, pT, tol, MAX)
    
    # iniPar and available modes
    # 'exp' - parameters a and l for a*np.exp(-x/l)
    # 'points' - direct list of dN [cm^-3] points as csv file read using get_initpoints()
    init_mode = "points"
    a  = 1e18/(1e7)**3                        # Amplitude
    l  = 100                                  # Length scale [nm]
    #iniPar = np.array([[a, l]])
    iniPar = get_initpoints("inits.csv")

    # This code follows a strict order of parameters:
    # matPar = [N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda]
    param_names = ["n0", "p0", "mu_n", "mu_p", "B", "Sf", "Sb", "tau_n", "tau_p", "rel. permitivity^-1"]
    unit_conversions = np.array([(1e7)**-3,(1e7)**-3,(1e7)**2/(1e9)*.02569257,(1e7)**2/(1e9)*.02569257,(1e7)**3/(1e9),(1e7)/(1e9),(1e7)/(1e9),1,1,lambda0])
    do_log = np.array([1,1,0,0,1,1,1,0,0,0])

    GPU_GROUP_SIZE = 16 ** 3                  # Number of simulations assigned to GPU at a time - GPU has limited memory
    ref1 = np.array([1,1,1,1,32,32,1,32,32,1])
    ref2 = np.array([1,1,1,1,16,16,1,16,16,1])
    ref4 = np.array([1,1,1,1,16,16,1,16,1,1])
    ref3 = np.array([1,16,1,1,16,16,1,16,16,1])
    refs = np.array([ref2])#, ref2, ref3])                         # Refinements
    

    minX = np.array([1e8, 1e15, 10, 10, 1e-11, 1e3, 1e-6, 1, 1, 13.6**-1])                        # Smallest param v$
    maxX = np.array([1e8, 1e15, 10, 10, 1e-9, 2e5, 1e-6, 100, 100, 13.6**-1])

    LOG_PL = True
    bval = 1e-10
    include_neighbors = True
    P_thr = float(np.prod(refs[0])) ** -1 * 2                 # Threshold P
    minP = np.array([0] + [P_thr for i in range(len(refs) - 1)])

    N    = np.array([0])                              # Initial N
    P    = np.array([1.0])                            # Initial P

    wdir = r"/blue/c.hages/cfai2304/"
    experimental_data_filename = sys.argv[1]
    out_filename = sys.argv[2]
    
    # Pre-checks
    try:
        num_params = len(param_names)
        assert (len(unit_conversions) == num_params), "Unit conversion array is missing entries"
        assert (len(do_log) == num_params), "do_log mask is missing values"
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
        print("Output: {}".format(out_filename))
        try:
            has_GPU = cuda.detect()
        except Exception:
            has_GPU = False

        if has_GPU: 
            device = cuda.get_current_device()
            num_SMs = getattr(device, "MULTIPROCESSOR_COUNT")
            TPB = 2 ** 7
            from pvSimPCR import pvSim
        else:
            print("No GPU detected - reverting to CPU simulation")
            num_SMs = -1
            from pvSim import pvSim

        #_continue = input("Continue? (y/n)")
        
        #if not (_continue == 'y'): raise KeyboardInterrupt("Aborted")
        
    except Exception as oops:
        print(oops)
        sys.exit(0)
        
    minX *= unit_conversions
    maxX *= unit_conversions

    import time
    clock0 = time.time()
    N, P = bayes(pvSim, N, P, refs, minX, maxX, iniPar, simPar, minP, e_data)
    print("Bayesim took {} s".format(time.time() - clock0))
    try:
        print("Writing to /blue:")
        marP = marginalP(N, P, refs)
        export_marginal_P(marP, np.prod(refs,axis=0), minX * (unit_conversions ** -1), maxX * (unit_conversions ** -1), param_names)
        cov_P(N, P, refs, minX * (unit_conversions ** -1), maxX * (unit_conversions ** -1))
    except:
        print("Write failed; rewriting to backup location /home:")
        wdir = r"/home/cfai2304/super_bayes/"
        marP = marginalP(N, P, refs)
        export_marginal_P(marP, np.prod(refs,axis=0), minX * (unit_conversions ** -1), maxX * (unit_conversions ** -1), param_names)
        cov_P(N, P, refs, minX * (unit_conversions ** -1), maxX * (unit_conversions ** -1))

    maxP(N, P, refs, minX *(unit_conversions ** -1) , maxX * (unit_conversions ** -1))
