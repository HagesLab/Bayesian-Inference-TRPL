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
        wheremax = np.unravel_index(np.argmax(P[ti], axis=None), P[ti].shape)
        print(X[wheremax[0]])
        print("Mag_offset: ", mag_grid[wheremax[1]])
        print("P = {}".format(np.max(P[ti])))
    np.save("{}_BAYRAN_MAX.npy".format(wdir + out_filename), np.append(X[wheremax[0]], mag_grid[wheremax[1]]))

def export_random_marP(X, P, mag_grid):
    np.save("{}_BAYRAN_X.npy".format(wdir + out_filename), X)
    if len(mag_grid) > 1:
        np.save("{}_BAYRAN_MAGS.npy".format(wdir + out_filename), mag_grid)
        bin_ctx = 48
        bin_cty = len(mag_grid)
        bx = np.arange(bin_ctx+1)
        by = np.arange(bin_cty+1)
        if do_log[4]:
            bx = minX[4] * (maxX[4]/minX[4])**((bx)/bin_ctx)
                            
        else:
            bx = minX[4] + (maxX[4]-minX[4])*(bx)/bin_ctx    # Get params

        by = mag_grid[0] + (mag_grid[-1]-mag_grid[0])*(by)/bin_cty

    P_over_mags = np.sum(P, axis=2)
    #P_over_mags = P[:, :, len(mag_grid) // 2]
    for ti, tf in enumerate(T_FACTORS):
        Pti = P_over_mags[ti]
        np.save("{}_BAYRAN_{}.npy".format(wdir + out_filename, int(tf)), Pti)

        # Only need B-mag_offset covariance
        if len(mag_grid) > 1:
            Pti = P[ti].flatten(order='F')
            Bm = np.tile(X[:, 4], len(mag_grid))
            m = np.repeat(mag_grid, len(X))
            h, bx, by = np.histogram2d(Bm, m, bins=[bx, by], weights=Pti)
            np.save("{}_BAYRAN_BMO_{}.npy".format(wdir + out_filename, int(tf)), h)
            np.save("{}_BAYRAN_MAG_{}.npy".format(wdir + out_filename, int(tf)), np.sum(P[ti], axis=0))
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
    P = np.zeros((len(T_FACTORS), len(N),  len(mag_grid)))                               # Likelihoods
    if OVERRIDE_EQUAL_MU:
        X[:,2] = X[:,3]

    return N, P, X

def simulate(model, data, nref, P, X, X_old, minus_err_sq, err_old, timepoints_per_ic, num_curves,
             sim_params, init_params, T_FACTORS, gpu_id, num_gpus, solver_time, err_sq_time, misc_time):
    try:
        cuda.select_device(gpu_id)
    except IndexError:
        print("Error: threads failed to launch")
        sys.exit()
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
                    solver_time[gpu_id] += model(plI, plN, plP, plE, X[blk:blk+size], 
                                                 sim_params, init_params[ic_num], 
                                                 TPB,8*num_SMs, max_sims_per_block, init_mode=init_mode)
                else:
                    plI = model(X[blk:blk+size], sim_params, init_params[ic_num])[1][-1]
        

                if LOG_PL:
                    #plI[plI<10*sys.float_info.min] = 10*sys.float_info.min
                    #plI = np.log10(plI)
                    misc_time[gpu_id] += fastlog(plI, 10*sys.float_info.min, TPB[0], num_SMs)

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
            err_sq_time[gpu_id] += prob(P[:, blk:blk+size], plI, values, mag_grid, bval_cutoff, T_FACTORS, TPB[0], num_SMs)
        # END LOOP OVER BLOCKS
    # END LOOP OVER ICs

    return

def select_accept(nref,P, P_old, minus_err_sq, err_old, X, X_old):
    # Evaluate acceptance criteria on the basis of squared errors - default temperature and zero mag offset
    minus_err_sq = P[-1, :, len(mag_grid) // 2]
    if nref == 0:
        accept = np.ones_like(minus_err_sq)

    else:
        # Less negative is more probable
        accept = np.where(minus_err_sq > err_old, 1, 0)

    print("Accept fraction: {}".format(np.sum(accept) / NUM_POINTS))

    for x in np.where(accept)[0]:
        err_old[x] = minus_err_sq[x]
        X_old[x] = X[x]
        P_old[:,x,:] = P[:,x,:]

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
    T_FACTORS = np.geomspace(len(data[0]) / 10, 10, 8)
    #T_FACTORS = np.array([100])
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
            bval_cutoff = np.mean(uncertainty)
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
    Time = 2000
    #Length = 2000
    Length  = 311                            # Length (nm)
    lambda0 = 704.3                           # q^2/(eps0*k_B T=25C) [nm]
    L   = 2 ** 7                                # Spatial points
    #T   = 4000
    T = 80000
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
    param_names = ["n0", "p0", "mun", "mup", "B", "Sf", "Sb", "taun", "taup", "lambda"]
    unit_conversions = np.array([(1e7)**-3,(1e7)**-3,(1e7)**2/(1e9)*.02569257,(1e7)**2/(1e9)*.02569257,(1e7)**3/(1e9),(1e7)/(1e9),(1e7)/(1e9),1,1,lambda0])
    do_log = np.array([1,1,0,0,1,1,1,0,0,0])

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

    minX = np.array([1e8, 1e8, 20, 1e-10, 4.8e-11, 1e-6, 10, 1, 1, 10**-1])                        # Smallest param v$
    maxX = np.array([1e8, 1e18, 20, 100, 4.8e-11, 5e2, 10, 1500, 1500, 10**-1])
    #minX = np.array([1e8, 1e15, 10, 10, 1e-11, 1e3, 1e-6, 1, 1, 10**-1])
    #maxX = np.array([1e8, 1e15, 10, 10, 1e-9, 2e5, 1e-6, 100, 100, 10**-1])
    #minX = np.array([1e8, 3e15, 20,20, 1e-11, 1e-6, 10, 511, 871, 10**-1])
    #maxX = np.array([1e8, 3e15, 20,20, 1e-9, 5e2, 10, 511, 871, 10**-1])
    mag_scale = (-1,1)
    mag_points = 31
    mag_grid = np.linspace(mag_scale[0], mag_scale[1], mag_points)
    #mag_grid = [0]

    LOADIN_PL = "load" in sys.argv[4]
    OVERRIDE_EQUAL_MU = True
    LOG_PL = True

    np.random.seed(420)
    RANDOM_SAMPLE = True
    NUM_POINTS = 2 ** 23

    scale_f = 1e-23 # [phot/cm^2 s] to [phot/nm^2 ns]
    sample_factor = 1
    data_is_noisy = True
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
        assert all(minX > 0), "Invalid param values"
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
        print("Equal mu override: {}".format(OVERRIDE_EQUAL_MU))
        for i in range(num_params):
            if minX[i] == maxX[i]:
                print("{}: {}".format(param_names[i], minX[i]))
                
            else:
                print("{}: {} to {} {}".format(param_names[i], minX[i], maxX[i], "log" if do_log[i] else "linear"))
        if not RANDOM_SAMPLE:
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
        export_random_marP(X, P, mag_grid)
    except Exception as e:
        print(e)
        print("Write failed; rewriting to backup location /home:")
        wdir = r"/home/cfai2304/super_bayes/"
        export_random_marP(X, P)

    maxP(N, P, X, refs, minX, maxX)
    print("Export took {}".format(time.time() - clock0))
