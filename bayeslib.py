#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:33:18 2020
@author: tladd
"""
from numba import cuda
import threading
import sys
import numpy as np

from probs import prob, fastlog
from bayes_io import save_raw_pl, load_raw_pl
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

def random_grid(minX, maxX, do_log, num_points, do_grid=False, refs=None):
    num_params = len(minX)
    grid = np.empty((num_points, num_params))
    
    for i in range(num_params):
        if minX[i] == maxX[i]:
            grid[:,i] = minX[i]
        else:
            #if do_grid:
            #    ind = np.arange(refs[i])
            #    if do_log[i]:
            #        possible_vals = minX[i] * (maxX[i]/minX[i]) ** ((ind+0.5) / refs[i])
            #    else:
            #        possible_vals = minX[i] + (maxX[i]-minX[i]) * (ind+0.5) / refs[i]
            #    grid[:,i] = np.random.choice(possible_vals, size=(len(grid[:,i],)))
            #else:
            if do_log[i]:
                grid[:,i] = 10 ** np.random.uniform(np.log10(minX[i]), np.log10(maxX[i]), (num_points,))
            else:
                grid[:,i] = np.random.uniform(minX[i], maxX[i], (num_points,))
            
    return grid


def make_grid(N, P, minX, maxX, do_log, sim_flags, nref=None, minP=None, refs=None):
    OVERRIDE_EQUAL_MU = sim_flags["override_equal_mu"]
    OVERRIDE_EQUAL_S = sim_flags["override_equal_s"]
    RANDOM_SAMPLE = sim_flags["random_sample"]
    NUM_POINTS = sim_flags["num_points"]

    if RANDOM_SAMPLE:
        N = np.arange(NUM_POINTS)
        X = random_grid(minX, maxX, do_log, NUM_POINTS, do_grid=False, refs=refs)
        print(len(X), "random points")

    else:
        #if P is not None:
        #    N   = N[np.where(np.sum(np.mean(P, axis=0), axis=1) > minP[nref])] # P < minP - avg over tfs, sum over mag
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
    P = np.zeros((len(N)))                               # Likelihoods
    if OVERRIDE_EQUAL_MU:
        X[:,2] = X[:,3]

    if OVERRIDE_EQUAL_S:
        X[:,6] = X[:,5]
    return N, P, X

def simulate(model, e_data, P, X, plI, num_curves,
             sim_params, init_params, sim_flags, gpu_info, gpu_id, solver_time, err_sq_time, misc_time):

    has_GPU = gpu_info["has_GPU"]
    GPU_GROUP_SIZE = gpu_info["sims_per_gpu"]
    num_gpus = gpu_info["num_gpus"]
    TPB = gpu_info["threads_per_block"]
    max_sims_per_block = gpu_info["max_sims_per_block"]

    LOADIN_PL = sim_flags["load_PL_from_file"]
    LOG_PL = sim_flags["log_pl"]
    NORMALIZE = sim_flags["self_normalize"]
    try:
        cuda.select_device(gpu_id)
    except IndexError:
        print("Error: threads failed to launch")
        return
    device = cuda.get_current_device()
    num_SMs = getattr(device, "MULTIPROCESSOR_COUNT")

    if isinstance(sim_params[0], (int, float)):
        thicknesses = [sim_params[0] for ic_num in range(num_curves)]
    elif isinstance(sim_params[0], list):
        thicknesses = list(sim_params[0])
    

    for ic_num in range(num_curves):
        times = e_data[0][ic_num]
        values = e_data[1][ic_num]
        std = e_data[2][ic_num]
        assert times[0] == 0, "Error: model time grid mismatch; times started with {} for ic {}".format(times[0], ic_num)

        # Update thickness
        sim_params[0] = thicknesses[ic_num]
        if gpu_id == 0: 
            print("new thickness: {}".format(sim_params[0]))

        num_observations = len(values)
        num_tsteps_needed = (num_observations-1)*sim_params[4]
        sim_params[3] = num_tsteps_needed
        sim_params[1] = times[-1]
        sim_params[5] = tuple(np.array(sim_params[5])*sim_params[4]*sim_params[3]//100)

        if gpu_id == 0: 
            print("Starting with values :{} \ncount: {}".format(values, len(values)))
            print("Taking {} timesteps".format(sim_params[3]))
            print("Final time: {}".format(sim_params[1]))

            print(sim_params)

        for blk in range(gpu_id*GPU_GROUP_SIZE,len(X),num_gpus*GPU_GROUP_SIZE):
            size = min(GPU_GROUP_SIZE, len(X) - blk)

            if not LOADIN_PL:
                plI[gpu_id] = np.empty((size, num_observations), dtype=np.float32) #f32 or f64 doesn't matter much here
                assert len(plI[gpu_id][0]) == len(values), "Error: plI size mismatch"
                plN = np.empty((size, 2, sim_params[2]))
                plP = np.empty((size, 2, sim_params[2]))
                plE = np.empty((size, 2, sim_params[2]+1))

                if has_GPU:
                    solver_time[gpu_id] += model(plI[gpu_id], plN, plP, plE, X[blk:blk+size, :-1], 
                                                 sim_params, init_params[ic_num],
                                                 TPB,8*num_SMs, max_sims_per_block, init_mode="points")
                else:
                    plI = model(X[blk:blk+size], sim_params, init_params[ic_num])[1][-1]
        
                if NORMALIZE:
                    # Normalize each model to its own t=0
                    plI[gpu_id] = plI[gpu_id].T
                    plI[gpu_id] /= plI[gpu_id][0]
                    plI[gpu_id] = plI[gpu_id].T
                if LOG_PL:
                    misc_time[gpu_id] += fastlog(plI[gpu_id], sys.float_info.min, TPB[0], num_SMs)
                if "+" in sys.argv[4]:
                    save_raw_pl(out_filename, ic_num, blk, plI[0])
                    

            else:
                print("Loading plI group {}".format(blk))
                plI = load_raw_pl(out_filename, ic_num, blk)

            # Calculate errors
            err_sq_time[gpu_id] += prob(P[blk:blk+size], plI[gpu_id], values, std, np.ascontiguousarray(X[blk:blk+size, -1]), 
                                        TPB[0], num_SMs)
        # END LOOP OVER BLOCKS
    # END LOOP OVER ICs

    return

def bayes(model, N, P, minX, maxX, do_log, init_params, sim_params, e_data, sim_flags, gpu_info):        # Driver function
    num_gpus = gpu_info["num_gpus"]
    solver_time = np.zeros(num_gpus)
    err_sq_time = np.zeros(num_gpus)
    misc_time = np.zeros(num_gpus)

    num_curves = len(init_params)
    #timepoints_per_ic = sim_params[3] // sim_params[4] + 1
    #for i in range(len(e_data[0])):
    #    assert (len(e_data[0][i]) % timepoints_per_ic == 0), "Error: experiment {} data length not a multiple of points_per_ic".format(i+1)

    N, P, X = make_grid(N, P, minX, maxX, do_log, sim_flags)

    sim_params = [list(sim_params) for i in range(num_gpus)]

    threads = []
    plI = [None for i in range(num_gpus)]
    for gpu_id in range(num_gpus):
        print("Starting thread {}".format(gpu_id))
        thread = threading.Thread(target=simulate, args=(model, e_data, P, X, plI,
                                  num_curves,sim_params[gpu_id], init_params, sim_flags, gpu_info, gpu_id,
                                  solver_time, err_sq_time, misc_time))
        threads.append(thread)
        thread.start()

    for gpu_id, thread in enumerate(threads):
        print("Ending thread {}".format(gpu_id))
        thread.join()
        print("Thread {} closed".format(gpu_id))

    print("Total tEvol time: {}, avg {}".format(solver_time, np.mean(solver_time)))
    print("Total err_sq time (temperatures and mag_offsets): {}, avg {}".format(err_sq_time, np.mean(err_sq_time)))
    print("Total misc time: {}, avg {}".format(misc_time, np.mean(misc_time)))
    return N, P, X


#-----------------------------------------------------------------------------#