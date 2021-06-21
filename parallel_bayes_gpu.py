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
    wheremax = np.unravel_index(np.argmax(P, axis=None), P.shape)
    print(X[wheremax[0]])
    print("Mag_offset: ", mag_grid[wheremax[1]])
    print("P = {}".format(P[wheremax]))

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

        if len(mag_grid) > 1:
            pID_1 = iterables[q]
            cov_P = np.zeros((pN[pID_1], len(mag_grid)))

            for i in np.unique(ind[:,pID_1]):
                cov_P[i] = np.sum(P[np.where(ind[:, pID_1] == i)], axis=0)

            m = pID_1
            im = np.array([*range(pN[m])]) + 0.5                 # Coords
            X1   = minX[m] + (maxX[m]-minX[m])*(im)/pN[m]    # Get params
            X2   = minX[m] * (maxX[m]/minX[m])**(im/pN[m])
            X = X1 * (1 - do_log[m]) + X2 * do_log[m]


            cov_P = np.hstack((X.reshape((len(X), 1)), cov_P))
            X = np.append(np.array([-1]), mag_grid, axis=0)
            cov_P = np.vstack((X.reshape((1, len(X))), cov_P))
            print("Writing covariance file {} and mag_offset".format(param_names[pID_1]))
            np.save(wdir + out_filename + "_BAYRES_" + param_names[pID_1] + "-mag_offset.npy", cov_P)
    return
        
def export_marginal_P(marP, pN, minX, maxX, param_names):

    for m in np.where(pN > 1)[0]:
        print("Writing marginal {} file".format(param_names[m]))
        im = np.array([*range(pN[m])]) + 0.5                 # Coords
        #X = minX[m] + (maxX[m]-minX[m])*im/pN[m]             # Param values
        X_lin   = minX[m] + (maxX[m]-minX[m])*(im)/pN[m]    # Get params
        X_log = minX[m] * (maxX[m]/minX[m])**((im)/pN[m])
        X =  X_lin * (1-do_log[m]) + X_log * do_log[m]

        print(np.vstack((X, marP[m])).T)
        np.savetxt(wdir + out_filename + "_BAYRES_" + param_names[m] + ".csv", np.vstack((X, marP[m])).T, delimiter=",")
    return        

def export_magsum(P):
    if len(mag_grid) > 1:
        sum_by_mag = np.sum(P, axis=0)
        print(np.vstack((mag_grid, sum_by_mag)).T)
        np.savetxt(wdir + out_filename + "_BAYRES_" + "mag_offset.csv", np.vstack((mag_grid, sum_by_mag)).T, delimiter=",")
    return

def make_grid(N, P, nref, refs, minX, maxX, minP, num_obs):
    N   = N[np.where(P > minP[nref])]                    # P < minP
    N   = refineGrid(N, refs[nref])                      # Refine grid
    Np  = np.prod(refs[nref])                            # Params per set
    P = np.zeros((len(N),  len(mag_grid)))                               # Likelihoods
    print("ref level, N: ", nref, len(N))

    X = np.empty((len(N), len(refs[0])))
        
    # TODO: Determine block size from GPU info instead of refinement?
    # Np cannot be modified! indexGrid assumes a certain value of Np
    for n in range(0, len(N), Np):                       # Loop over blks
        Nn  = N[n:n+Np]                                  # Cells block
        ind = indexGrid(Nn,  refs[0:nref+1])             # Get coordinates
        #X   = paramGrid(ind, refs[0:nref+1], minX, maxX) # Get params
        X[n:n+Np] = paramGrid(ind, refs[0:nref+1], minX, maxX) # Get params
    return N, P, X

def simulate(model, P, ic_num, values, X, timepoints_per_ic, 
             sim_params, init_params, T_FACTOR, gpu_id, num_gpus):
    cuda.select_device(gpu_id)
    device = cuda.get_current_device()
    num_SMs = getattr(device, "MULTIPROCESSOR_COUNT")
    
    for blk in range(gpu_id*GPU_GROUP_SIZE,len(X),num_gpus*GPU_GROUP_SIZE):
        size = min(GPU_GROUP_SIZE, len(X) - blk)

        if not LOADIN_PL:
            plI = np.empty((size, timepoints_per_ic), dtype=np.float32)
            plN = np.empty((size, 2, sim_params[2]))
            plP = np.empty((size, 2, sim_params[2]))
            plE = np.empty((size, 2, sim_params[2]+1))

            if has_GPU:
                model(plI, plN, plP, 
                      plE, X[blk:blk+size], sim_params, init_params[ic_num], 
                      TPB, num_SMs, max_sims_per_block, init_mode=init_mode)
            else:
                plI = model(X[blk:blk+size], sim_params, init_params[ic_num])[1][-1]
        

            if LOG_PL:         
                plI[plI<10*sys.float_info.min] = 10*sys.float_info.min
                plI = np.log10(plI)

            try:
                np.save("{}{}plI{}_grp{}.npy".format(wdir,out_filename, ic_num, blk), plI)
                print("Saved plI of size ", plI.shape)
            except Exception as e:
                print("Warning: save failed\n", e)

        else:
            print("Loading plI{} group {}".format(ic_num,blk))
            try:
                plI = np.load("{}{}plI{}_grp{}.npy".format(wdir,out_filename,ic_num, blk))
                print("Loaded plI of size ", plI.shape)
            except Exception as e:
                print("Error: load failed\n", e)
                sys.exit()


        # Calculate errors
        v_dev = cuda.to_device(values)
        m_dev = cuda.to_device(mag_grid)
        plI_dev = cuda.to_device(plI)
        P_dev = cuda.to_device(np.zeros_like(P[blk:blk+size]))
        kernel_lnP[num_SMs, TPB](P_dev, plI_dev, v_dev, m_dev, bval_cutoff, T_FACTOR)
        cuda.synchronize()
        P[blk:blk+size] += P_dev.copy_to_host()
    return

def normalize(P):
    # Normalization scheme - to ensure that np.sum(P) is never zero due to mass underflow
    # First, shift lnP's up so max lnP is zero, ensuring at least one nonzero P
    # Then shift lnP a little further to maximize number of non-underflowing values
    # without causing overflow
    # Key is only to add or subtract from lnP - that way any introduced factors cancel out
    # during normalize by sum(P)
    P = np.exp(P - np.max(P) + 1000*np.log(2) - np.log(P.size))
    P  /= np.sum(P)                                      # Normalize P's
    return P

def bayes(model, N, P, refs, minX, maxX, init_params, sim_params, minP, data):        # Driver function
    global num_SMs
    global has_GPU
    global init_mode
    global GPU_GROUP_SIZE
    for nref in range(len(refs)):                            # Loop refinements
        num_curves = len(init_params)
        timepoints_per_ic = sim_params[3] // sim_params[4] + 1
        assert (len(data[0]) % timepoints_per_ic == 0), "Error: exp data length not a multiple of points_per_ic"

        N, P, X = make_grid(N, P, nref, refs, minX, maxX, minP, num_curves*timepoints_per_ic)

        ## OVERRIDE: MAKE SRH TAUS EQUAL
        #X[:,8] = X[:,7]
        if OVERRIDE_EQUAL_MU:
            X[:,2] = X[:,3]
        for ic_num in range(num_curves):
            times = data[0][ic_num*timepoints_per_ic:(ic_num+1)*timepoints_per_ic]
            values = data[1][ic_num*timepoints_per_ic:(ic_num+1)*timepoints_per_ic]
            std = data[2][ic_num*timepoints_per_ic:(ic_num+1)*timepoints_per_ic]
            assert times[0] == 0, "Error: model time grid mismatch; times started with {} for ic {}".format(times[0], ic_num)
            try:
                T_FACTOR = float(sys.argv[5]) 
            except Exception:
                T_FACTOR = len(values)

            print("Temperature factor: ", str(T_FACTOR), "T=", str(len(values) / T_FACTOR))
            print("values", values)

            num_gpus = 4
            threads = []
            for gpu_id in range(num_gpus):
                print("Starting thread {}".format(gpu_id))
                thread = threading.Thread(target=simulate, args=(model, P, ic_num, values, X, 
                                          timepoints_per_ic, sim_params, init_params, T_FACTOR, gpu_id, num_gpus))
                threads.append(thread)
                thread.start()

            for gpu_id, thread in enumerate(threads):
                print("Ending thread {}".format(gpu_id))
                thread.join()
                print("Thread {} closed".format(gpu_id))

        P = normalize(P)
    return N, P


#-----------------------------------------------------------------------------#
import csv
from numba import cuda
import threading
import sys
def get_data(exp_file, scale_f=1, sample_f=1, noisy=False):
    with open(exp_file, newline='') as file:
        ifstream = csv.reader(file)
        t = []
        PL = []
        uncertainty = []
        count = 0
        for row in ifstream:
            if float(row[0]) == 0:
                count = 0
            if not (count % sample_f):
                t.append(float(row[0]))
                PL.append(float(row[1]))
                uncertainty.append(float(row[2]))
            
            count += 1

    t = np.array(t)
    PL = np.array(PL) * scale_f
    uncertainty = np.array(uncertainty) * scale_f
  
    if LOG_PL:
        if noisy:
            # Assume const uncertainty, set minimum observable value to that
            global bval_cutoff
            PL[PL < bval_cutoff] = bval_cutoff
        else:
            # Set minimum observable to 1
            PL[PL < scale_f] = scale_f
        uncertainty /= PL
        PL = np.log10(PL)
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
    #Time    = 250                                 # Final time (ns)
    #Length = 2000
    Time    = 2000
    Length  = 311                            # Length (nm)
    lambda0 = 704.3                           # q^2/(eps0*k_B T=25C) [nm]
    L   = 2 ** 7                                # Spatial points
    #T   = 2000
    T   = 80000                                # Time points
    plT = 1                                  # Set PL interval (dt)
    pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
    tol = 5                                   # Convergence tolerance
    MAX = 1000                                  # Max iterations
    pT = tuple(np.array(pT)*T//100)
    simPar = (Length, Time, L, T, plT, pT, tol, MAX)
    
    # iniPar and available modes
    # 'exp' - parameters a and l for a*np.exp(-x/l)
    # 'points' - direct list of dN [cm^-3] points as csv file read using get_initpoints()
    init_mode = "points"
    a  = 1e18/(1e7)**3                        # Amplitude
    l  = 100                                  # Length scale [nm]
    #iniPar = np.array([[a, l]])
    iniPar = get_initpoints(sys.argv[2])

    # This code follows a strict order of parameters:
    # matPar = [N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda]
    param_names = ["n0", "p0", "mu_n", "mu_p", "B", "Sf", "Sb", "tau_n", "tau_p", "rel. permitivity^-1"]
    unit_conversions = np.array([(1e7)**-3,(1e7)**-3,(1e7)**2/(1e9)*.02569257,(1e7)**2/(1e9)*.02569257,(1e7)**3/(1e9),(1e7)/(1e9),(1e7)/(1e9),1,1,lambda0])
    do_log = np.array([1,1,0,0,1,1,1,0,0,0])

    GPU_GROUP_SIZE = 2 ** 12                  # Number of simulations assigned to GPU at a time - GPU has limited memory
    ref1 = np.array([1,6,1,4,6,6,1,6,6,1])
    ref2 = np.array([1,1,1,1,16,16,1,16,16,1])
    ref4 = np.array([1,1,1,1,1,1,1,32,1,1])
    ref3 = np.array([1,1,1,8,8,1,1,8,8,1])
    ref5 = np.array([1,2,1,6,6,6,1,6,6,1])
    refs = np.array([ref1])#, ref2, ref3])                         # Refinements
    

    minX = np.array([1e8, 1e13, 20, 1, 1e-11, 1e-1, 10, 1, 1, 10**-1])                        # Smallest param v$
    maxX = np.array([1e8, 1e17, 20, 100, 1e-9, 1e5, 10, 1000, 1000, 10**-1])
    #minX = np.array([1e8, 1e15, 10, 10, 1e-11, 1e3, 1e-6, 1, 1, 10**-1])
    #maxX = np.array([1e8, 1e15, 10, 10, 1e-9, 2e5, 1e-6, 100, 100, 10**-1])
    #minX = np.array([1e8, 3e15, 20, 20, 4.8e-11, 10, 10, 1, 871, 10**-1])
    #maxX = np.array([1e8, 3e15, 20, 20, 4.8e-11, 10, 10, 1000, 871, 10**-1])
    mag_scale = (-2,2)
    mag_points = 31
    mag_grid = np.linspace(mag_scale[0], mag_scale[1], mag_points)
    #mag_grid = [0]

    LOADIN_PL = sys.argv[4] != "new"
    OVERRIDE_EQUAL_MU = True
    LOG_PL = True
    scale_f = 1e-23 # [phot/cm^2 s] to [phot/nm^2 ns]
    sample_factor = 1
    data_is_noisy = False
    bval_cutoff = 1 * scale_f
    P_thr = float(np.prod(refs[0])) ** -1 * 2                 # Threshold P
    minP = np.array([0] + [P_thr for i in range(len(refs) - 1)])

    N    = np.array([0])                              # Initial N
    P    = np.array([1.0])                            # Initial P

    wdir = r"/blue/c.hages/cfai2304/"
    experimental_data_filename = sys.argv[1]
    out_filename = sys.argv[3]
    
    # Pre-checks
    try:
        num_params = len(param_names)
        assert (len(unit_conversions) == num_params), "Unit conversion array is missing entries"
        assert (len(do_log) == num_params), "do_log mask is missing values"
        assert (len(minX) == num_params), "Missing min param values"
        assert (len(maxX) == num_params), "Missing max param values"  
        assert all(minX > 0), "Invalid param values"
        assert all(minX <= maxX), "Min params larger than max params"
        if OVERRIDE_EQUAL_MU:
            for ref in refs:
                assert ref[2] == 1, "Equal mu override is on but mu_n is being subdivided"

        for i in range(len(refs[0])):
            if not refs[0,i] == 1:
                assert not (minX[i] == maxX[i]), "{} is subdivided but min val == max val".format(param_names[i])
            else:
                assert (minX[i] == maxX[i]), "{} is not subdivided but min val != max val".format(param_names[i])
        # TODO: Additional checks involving refs
            
        print("Starting simulations with the following parameters:")
        print("Equal mu override: {}".format(OVERRIDE_EQUAL_MU))
        for i in range(num_params):
            if minX[i] == maxX[i]:
                print("{}: {}".format(param_names[i], minX[i]))
                
            else:
                print("{}: {} to {}".format(param_names[i], minX[i], maxX[i]))
        
        print("Refinement levels:")
        for i in range(num_params):
            print("{}: {}".format(param_names[i], refs[:,i]))        
        e_data = get_data(experimental_data_filename, scale_f=scale_f, sample_f = sample_factor, noisy=data_is_noisy) 
        print("\nExperimental data - {}".format(experimental_data_filename))
        print("Data considered noisy: {}".format(data_is_noisy))
        print("Sample factor: {}".format(sample_factor))
        print(e_data)
        print("Output: {}".format(out_filename))
        try:
            print("Detecting GPU...")
            has_GPU = cuda.detect()
        except Exception as e:
            print(e)
            has_GPU = False

        if has_GPU: 
            device = cuda.get_current_device()
            num_SMs = getattr(device, "MULTIPROCESSOR_COUNT")
            TPB = (2 ** 7,)
            max_sims_per_block = 3           # Maximum of 6 due to shared memory limit
            from pvSimPCR import pvSim
            from probs import lnP, kernel_lnP
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
        export_magsum(P)
        cov_P(N, P, refs, minX * (unit_conversions ** -1), maxX * (unit_conversions ** -1))
    except:
        print("Write failed; rewriting to backup location /home:")
        wdir = r"/home/cfai2304/super_bayes/"
        marP = marginalP(N, P, refs)
        export_marginal_P(marP, np.prod(refs,axis=0), minX * (unit_conversions ** -1), maxX * (unit_conversions ** -1), param_names)
        export_magsum(P)
        cov_P(N, P, refs, minX * (unit_conversions ** -1), maxX * (unit_conversions ** -1))

    maxP(N, P, refs, minX *(unit_conversions ** -1) , maxX * (unit_conversions ** -1))
