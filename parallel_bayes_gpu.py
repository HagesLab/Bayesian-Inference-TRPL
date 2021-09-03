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

def random_grid(minX, maxX, num_points, do_grid=False, refs=None):
    num_params = len(minX)
    grid = np.empty((num_points, num_params))
    
    for i in range(num_params):
        if minX[i] == maxX[i]:
            grid[:,i] = minX[i]
        else:
            if do_grid:
                ind = np.arange(refs[i])
                if do_log[i]:
                    possible_vals = minX[i] * (maxX[i]/minX[i]) ** ((ind+0.5) / refs[i])
                else:
                    possible_vals = minX[i] + (maxX[i]-minX[i]) * (ind+0.5) / refs[i]
                grid[:,i] = np.random.choice(possible_vals, size=(len(grid[:,i],)))
            else:
                if do_log[i]:
                    grid[:,i] = 10 ** np.random.uniform(np.log10(minX[i]), np.log10(maxX[i]), (num_points,))
                else:
                    grid[:,i] = np.random.uniform(minX[i], maxX[i], (num_points,))
            
    return grid

def maxP(N,P,X, refs, minX, maxX):
    if not RANDOM_SAMPLE:
        pN = np.prod(refs, axis=0)
        ind = indexGrid(N,refs)
        X = paramGrid(ind, refs, minX, maxX)
    for ti, tf in enumerate(T_FACTORS):
        print("Temperature:", tf)
        wheremax = np.argmax(P[ti])
        print(X[wheremax])
        print("P = {}".format(P[ti, wheremax]))
    #np.save("{}_BAYRAN_MAX.npy".format(wdir + out_filename), X[wheremax])

def export_random_marP(X, P):
    np.save("{}_BAYRAN_X.npy".format(wdir + out_filename), X)

    for ti, tf in enumerate(T_FACTORS):
        Pti = P[ti]
        np.save("{}_BAYRAN_{}.npy".format(wdir + out_filename, int(tf)), Pti)
    return

def make_grid(N, P, nref, refs, minX, maxX, minP, num_obs):
    if RANDOM_SAMPLE:
        N = np.arange(NUM_POINTS)
        print(NUM_POINTS, "random points")
        X = random_grid(minX, maxX, NUM_POINTS, do_grid=False, refs=refs)

    else:
        if P is not None:
            N   = N[np.where(np.sum(np.mean(P, axis=0), axis=1) > minP[nref])] # P < minP - avg over tfs, sum over mag
        N   = refineGrid(N, refs[nref])                      # Refine grid
        Np  = np.prod(refs[nref])                            # Params per set
        print("ref level, N: ", nref, len(N))

        X = np.empty((len(N), len(refs[0])))
        
        # TODO: Determine block size from GPU info instead of refinement?
        # Np cannot be modified! indexGrid assumes a certain value of Np
        for n in range(0, len(N), Np):                       # Loop over blks
            Nn  = N[n:n+Np]                                  # Cells block
            ind = indexGrid(Nn,  refs[0:nref+1])             # Get coordinates
            #X   = paramGrid(ind, refs[0:nref+1], minX, maxX) # Get params
            X[n:n+Np] = paramGrid(ind, refs[0:nref+1], minX, maxX) # Get params
    P = np.zeros((len(T_FACTORS), len(N)))                               # Likelihoods
    if OVERRIDE_EQUAL_MU:
        X[:,2] = X[:,3]

    if OVERRIDE_EQUAL_S:
        X[:,6] = X[:,5]
    return N, P, X

def simulate(model, data, nref, P, X, X_old, minus_err_sq, err_old, timepoints_per_ic, num_curves,
             sim_params, init_params, T_FACTORS, gpu_id, num_gpus, solver_time, err_sq_time, misc_time):
    try:
        cuda.select_device(gpu_id)
    except IndexError:
        print("Error: threads failed to launch")
        return
    device = cuda.get_current_device()
    num_SMs = getattr(device, "MULTIPROCESSOR_COUNT")
    for ic_num in range(num_curves):
        times = data[0][ic_num*timepoints_per_ic:(ic_num+1)*timepoints_per_ic]
        values = data[1][ic_num*timepoints_per_ic:(ic_num+1)*timepoints_per_ic]
        std = data[2][ic_num*timepoints_per_ic:(ic_num+1)*timepoints_per_ic]
        assert times[0] == 0, "Error: model time grid mismatch; times started with {} for ic {}".format(times[0], ic_num)

        for blk in range(gpu_id*GPU_GROUP_SIZE,len(X),num_gpus*GPU_GROUP_SIZE):
            size = min(GPU_GROUP_SIZE, len(X) - blk)

            if not LOADIN_PL:
                plI = np.empty((size, timepoints_per_ic), dtype=np.float32)
                plN = np.empty((size, 2, sim_params[2]))
                plP = np.empty((size, 2, sim_params[2]))
                plE = np.empty((size, 2, sim_params[2]+1))

                if has_GPU:
                    solver_time[gpu_id] += model(plI, plN, plP, plE, X[blk:blk+size, :-1], 
                                                 sim_params, init_params[ic_num], 
                                                 TPB,8*num_SMs, max_sims_per_block, init_mode=init_mode)
                else:
                    plI = model(X[blk:blk+size], sim_params, init_params[ic_num])[1][-1]
        
                if NORMALIZE:
                    # Normalize each model to its own t=0
                    plI = plI.T
                    plI /= plI[0]
                    plI = plI.T
                if LOG_PL:
                    misc_time[gpu_id] += fastlog(plI, bval_cutoff, TPB[0], num_SMs)

                if "+" in sys.argv[4]:
                    try:
                        np.save("{}{}plI{}_grp{}.npy".format(wdir,out_filename,ic_num, blk), plI)
                        print("Saved plI of size ", plI.shape)
                    except Exception as e:
                        print("Warning: save failed\n", e)

            else:
                print("Loading plI group {}".format(blk))
                try:
                    plI = np.load("{}{}plI{}_grp{}.npy".format(wdir,out_filename, ic_num, blk))
                    print("Loaded plI of size ", plI.shape)
                except Exception as e:
                    print("Error: load failed\n", e)
                    sys.exit()


            # Calculate errors
            err_sq_time[gpu_id] += prob(P[:, blk:blk+size], plI, values, std, np.ascontiguousarray(X[blk:blk+size, -1]), 
                                        T_FACTORS, TPB[0], num_SMs)
        # END LOOP OVER BLOCKS
    # END LOOP OVER ICs

    return

def select_accept(nref,P, P_old, minus_err_sq, err_old, X, X_old):
    # Evaluate acceptance criteria on the basis of squared errors - default temperature and zero mag offset
    minus_err_sq = P[0, :]
    if nref == 0:
        accept = np.ones_like(minus_err_sq)

    else:
        # Less negative is more probable
        accept = np.where(minus_err_sq > err_old, 1, 0)

    print("Accept fraction: {}".format(np.sum(accept) / NUM_POINTS))

    for x in np.where(accept)[0]:
        err_old[x] = minus_err_sq[x]
        X_old[x] = X[x]
        P_old[:,x] = P[:,x]

    return

def normalize(P):
    # Normalization scheme - to ensure that np.sum(P) is never zero due to mass underflow
    # First, shift lnP's up so max lnP is zero, ensuring at least one nonzero P
    # Then shift lnP a little further to maximize number of non-underflowing values
    # without causing overflow
    # Key is only to add or subtract from lnP - that way any introduced factors cancel out
    # during normalize by sum(P)
    for ti, tf in enumerate(T_FACTORS):
        P[ti] = np.exp(P[ti] - np.max(P[ti]) + 1000*np.log(2) - np.log(P[ti].size))
        P[ti]  /= np.sum(P[ti])                                      # Normalize P's
    return P

def bayes(model, N, P, refs, minX, maxX, init_params, sim_params, minP, data):        # Driver function
    global num_SMs
    global has_GPU
    global init_mode
    global GPU_GROUP_SIZE
    global T_FACTORS
    global num_gpus

    solver_time = np.zeros(num_gpus)
    err_sq_time = np.zeros(num_gpus)
    misc_time = np.zeros(num_gpus)

    num_curves = len(init_params)
    timepoints_per_ic = sim_params[3] // sim_params[4] + 1
    assert (len(data[0]) % timepoints_per_ic == 0), "Error: exp data length not a multiple of points_per_ic"
    T_FACTORS = np.geomspace(len(data[0])/10, 10, 8)
    #T_FACTORS = np.geomspace(450, 10, 8)
    #T_FACTORS = np.array([1])
    print("Temperatures: ", T_FACTORS)

    N, P, X = make_grid(N, P, 0, refs, minX, maxX, minP, num_curves*timepoints_per_ic)
    minus_err_sq = np.zeros(len(N))
    err_old = np.zeros_like(minus_err_sq)
    P_old = np.zeros_like(P)
    X_old = np.zeros_like(X)
    accept = np.zeros_like(minus_err_sq)

    for nref in range(mc_refs):                            # Loop refinements

        threads = []
        for gpu_id in range(num_gpus):
            print("Starting thread {}".format(gpu_id))
            thread = threading.Thread(target=simulate, args=(model, data, nref, P, X, X_old, minus_err_sq, err_old,
                                      timepoints_per_ic, num_curves,sim_params, init_params, T_FACTORS, gpu_id, num_gpus,
                                      solver_time, err_sq_time, misc_time))
            threads.append(thread)
            thread.start()

        for gpu_id, thread in enumerate(threads):
            print("Ending thread {}".format(gpu_id))
            thread.join()
            print("Thread {} closed".format(gpu_id))

        select_accept(nref, P, P_old, minus_err_sq, err_old, X, X_old)

        #N, P, X = make_grid(N, P, nref+1, refs, minX, maxX, minP, num_curves*timepoints_per_ic)
        
        minus_err_sq = np.zeros(len(N))

    P_old = normalize(P_old)

    print("Total tEvol time: {}, avg {}".format(solver_time, np.mean(solver_time)))
    print("Total err_sq time (temperatures and mag_offsets): {}, avg {}".format(err_sq_time, np.mean(err_sq_time)))
    print("Total misc time: {}, avg {}".format(misc_time, np.mean(misc_time)))
    return N, P_old, X_old


#-----------------------------------------------------------------------------#
import csv
from numba import cuda
import threading
import sys
import time
def get_data(exp_file, scale_f=1, sample_f=1):
    global bval_cutoff
    with open(exp_file, newline='') as file:
        ifstream = csv.reader(file)
        t = []
        PL = []
        uncertainty = []
        count = 0
        dataset_end_inds = [0]
        for row in ifstream:
            if float(row[0]) == 0:
                dataset_end_inds.append(dataset_end_inds[-1] + count)
                count = 0
            if not (count % sample_f):
                t.append(float(row[0]))
                PL.append(float(row[1]))
                uncertainty.append(float(row[2]))
            
            count += 1
    # Unpack
    t = np.array(t)
    PL = np.array(PL) * scale_f
    uncertainty = np.array(uncertainty) * scale_f

    # FIgure out where curves start and end
    dataset_end_inds.append(dataset_end_inds[-1] + count)
    dataset_end_inds.pop(0)
    dataset_end_inds.pop(-1)
    dataset_end_inds.append(None)
  
    if NORMALIZE:
        print(dataset_end_inds)
        print("t=0 pl values:", PL[dataset_end_inds[:-1]])

        NORM_FACTORS = PL[dataset_end_inds[:-1]]

        # Normalize everything to its own t=0
        for i in range(len(dataset_end_inds[:-1])):
            PL[dataset_end_inds[i]:dataset_end_inds[i+1]] /= NORM_FACTORS[i]
            uncertainty[dataset_end_inds[i]:dataset_end_inds[i+1]] /= NORM_FACTORS[i]

    bval_cutoff = 10 * sys.float_info.min

    if LOG_PL:
        print("cutoff", bval_cutoff)
        print("Num exp points affected by cutoff", np.sum(PL < bval_cutoff))

        # Deal with noisy negative values before taking log
        #bval_cutoff = np.mean(uncertainty)
        PL = np.abs(PL)
        #print("pl:", PL)
        PL[PL < bval_cutoff] = bval_cutoff

        uncertainty /= PL
        uncertainty /= 2.3 # Since we use log10 instead of ln
        PL = np.log10(PL)
        print("uncertainty", uncertainty)
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
    Time = 2000
    #Time = 131867*0.025
    Time = 10
    #Length = 2000
    Length  = 311                            # Length (nm)
    lambda0 = 704.3                           # q^2/(eps0*k_B T=25C) [nm]
    L   = 2 ** 7                                # Spatial points
    #T   = 4000
    T = 80000
    T = 400
    #T = 131867
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
    # matPar = [N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda, mag_offset]
    param_names = ["n0", "p0", "mun", "mup", "B", "Sf", "Sb", "taun", "taup", "lambda", "mag_offset"]
    unit_conversions = np.array([(1e7)**-3,(1e7)**-3,(1e7)**2/(1e9)*.02569257,(1e7)**2/(1e9)*.02569257,(1e7)**3/(1e9),(1e7)/(1e9),(1e7)/(1e9),1,1,lambda0, 1])
    do_log = np.array([1,1,0,0,1,1,1,0,0,0,0])

    GPU_GROUP_SIZE = 2 ** 13                  # Number of simulations assigned to GPU at a time - GPU has limited memory
    num_gpus = 8

    ref1 = np.array([1,1,1,128,128,1,1,1,1,1])
    ref2 = np.array([1,24,1,1,24,24,1,24,24,1])
    ref3 = np.array([1,1,1,1,16,16,1,16,16,1])
    ref4 = np.array([1,1,1,1,1,1,1,32,1,1])
    #ref3 = np.array([1,1,1,8,8,1,1,8,8,1])
    ref5 = np.array([1,8,1,4,8,8,1,8,8,1])
    refs = np.array([ref1])#, ref2, ref3])                         # Refinements
    mc_refs = 1

    minX = np.array([1e8, 1e8, 20, 1e-4, 1e-11, 1e-4, 10, 1, 1, 10**-1, -1])
    maxX = np.array([1e8, 1e18, 20, 100, 1e-9, 5e2, 10, 1500, 1500, 10**-1, 1])
    #minX = np.array([1e8, 3e15, 20, 0, 4.8e-11, 1e-4, 10, 1, 1, 10**-1, 0])
    #maxX = np.array([1e8, 3e15, 20, 100, 4.8e-11, 5e2, 10, 1500, 1500, 10**-1, 0])
    #minX = np.array([1e8, 1e8, 20, 1e-10, 1e-11, 1, 1e4, 1, 1, 10**-1, -0.2])
    #maxX = np.array([1e8, 1e18, 20, 100, 1e-9, 1e5, 1e4, 1500, 1500, 10**-1, 0.2])
    #minX = np.array([1e8, 1e15, 10, 10, 1e-11, 1e3, 1e-6, 1, 1, 10**-1])
    #maxX = np.array([1e8, 1e15, 10, 10, 1e-9, 2e5, 1e-6, 100, 100, 10**-1])
    #minX = np.array([1e8, 3e15, 20,20, 1e-11, 1e-6, 10, 511, 871, 10**-1])
    #maxX = np.array([1e8, 3e15, 20,20, 1e-9, 5e2, 10, 511, 871, 10**-1])

    LOADIN_PL = "load" in sys.argv[4]
    OVERRIDE_EQUAL_MU = True
    OVERRIDE_EQUAL_S = False
    LOG_PL = True
    NORMALIZE = False

    np.random.seed(42)
    RANDOM_SAMPLE = True
    NUM_POINTS = 2 ** 20

    scale_f = 1e-23 # [phot/cm^2 s] to [phot/nm^2 ns]
    sample_factor = 1
    P_thr = float(np.prod(refs[0])) ** -1 * 2                 # Threshold P
    minP = np.array([0] + [P_thr for i in range(len(refs) - 1)])

    N    = np.array([0])                              # Initial N
    P    = None                            # Initial P

    experimental_data_filename = sys.argv[1]
    out_filename = sys.argv[3]
    wdir = r"/blue/c.hages/cfai2304/{}/".format(out_filename)
    
    # Pre-checks
    try:
        num_params = len(param_names)
        assert (len(unit_conversions) == num_params), "Unit conversion array is missing entries"
        assert (len(do_log) == num_params), "do_log mask is missing values"
        assert (len(minX) == num_params), "Missing min param values"
        assert (len(maxX) == num_params), "Missing max param values"  
        assert all(minX <= maxX), "Min params larger than max params"
        if OVERRIDE_EQUAL_MU:
            for ref in refs:
                assert ref[2] == 1, "Equal mu override is on but mu_n is being subdivided"

        if not RANDOM_SAMPLE:
            for i in range(len(refs[0])):
                if not refs[0,i] == 1:
                    assert not (minX[i] == maxX[i]), "{} is subdivided but min val == max val".format(param_names[i])
                else:
                    assert (minX[i] == maxX[i]), "{} is not subdivided but min val != max val".format(param_names[i])
        # TODO: Additional checks involving refs
            
        print("Starting simulations with the following parameters:")
        print("{} iterations".format(mc_refs))
        print("Log PL: {}".format(LOG_PL))
        print("Equal mu override: {}".format(OVERRIDE_EQUAL_MU))
        print("Equal Sf=Sb override: {}".format(OVERRIDE_EQUAL_S))
        print("Normalize all curves: {}".format(NORMALIZE))
        for i in range(num_params):
            if minX[i] == maxX[i]:
                print("{}: {}".format(param_names[i], minX[i]))
                
            else:
                print("{}: {} to {} {}".format(param_names[i], minX[i], maxX[i], "log" if do_log[i] else "linear"))
        if not RANDOM_SAMPLE:
            print("Refinement levels:")
            for i in range(num_params):
                print("{}: {}".format(param_names[i], refs[:,i]))        
        e_data = get_data(experimental_data_filename, scale_f=scale_f, sample_f = sample_factor) 
        print("\nExperimental data - {}".format(experimental_data_filename))
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
            from probs import prob, fastlog
        else:
            print("No GPU detected - reverting to CPU simulation")
            raise NotImplementedError
            num_SMs = -1
            from pvSim import pvSim

    except Exception as oops:
        print(oops)
        sys.exit(0)
        
    minX *= unit_conversions
    maxX *= unit_conversions

    import time
    clock0 = time.time()
    N, P, X = bayes(pvSim, N, P, refs, minX, maxX, iniPar, simPar, minP, e_data)
    print("Bayesim took {} s".format(time.time() - clock0))

    minX /= unit_conversions
    maxX /= unit_conversions
    X /= unit_conversions

    try:
        import os
        print("Creating dir {}".format(out_filename))
        os.mkdir(wdir)
    except FileExistsError:
        print("{} dir already exists".format(out_filename))
    #np.save("{}_BAYRAN_P.npy".format(wdir + out_filename), P)
    #np.save("{}_BAYRAN_X.npy".format(wdir + out_filename), X)

    clock0 = time.time()
    try:
        print("Writing to /blue:")
        export_random_marP(X, P)
    except Exception as e:
        print(e)
        print("Write failed; rewriting to backup location /home:")
        wdir = r"/home/cfai2304/super_bayes/"
        export_random_marP(X, P)

    maxP(N, P, X, refs, minX, maxX)
    print("Export took {}".format(time.time() - clock0))
