#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:33:18 2020
@author: tladd
Driver functions for inference algorithm - random sampler, simulation, likelihood calculation
"""
from numba import cuda
import threading
import sys
import numpy as np
import time
from scipy.interpolate import griddata

from probs import prob, fastlog
from bayes_io import save_raw_pl, load_raw_pl

def random_grid(minX, maxX, do_log, num_points, do_grid=False, refs=None):
    """ Draw [num_points] random points from hyperspace bounded by [minX], [maxX] """
    num_params = len(minX)
    grid = np.empty((num_points, num_params))
    
    for i in range(num_params):
        if minX[i] == maxX[i]:
            grid[:,i] = minX[i]
        else:
            if do_log[i]:
                grid[:,i] = 10 ** np.random.uniform(np.log10(minX[i]), np.log10(maxX[i]), (num_points,))
            else:
                grid[:,i] = np.random.uniform(minX[i], maxX[i], (num_points,))
            
    return grid

def make_grid(N, P, num_exp, minX, maxX, do_log, sim_flags, nref=None, minP=None, refs=None):
    """ Set up sampling grid - either random sample or (DEPRECATED) coarse grid sample """
    OVERRIDE_EQUAL_MU = sim_flags["override_equal_mu"]
    OVERRIDE_EQUAL_S = sim_flags["override_equal_s"]
    OVERRIDE_EQUAL_AUGER = sim_flags["override_equal_auger"]
    RANDOM_SAMPLE = sim_flags["random_sample"]
    NUM_POINTS = sim_flags["num_points"]

    if RANDOM_SAMPLE:
        N = np.arange(NUM_POINTS)
        X = random_grid(minX, maxX, do_log, NUM_POINTS, do_grid=False, refs=refs)

    else:
        raise NotImplementedError("Grid sample currently deprecated; set RANDOM_SAMPLE to true")
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
    P = np.zeros((num_exp, len(N)))                               # Likelihoods
    
    # FIXME: reference by name instead of index
    if OVERRIDE_EQUAL_MU:
        X[:,2] = X[:,3]

    if OVERRIDE_EQUAL_S:
        X[:,6] = X[:,5]
        
    if OVERRIDE_EQUAL_AUGER:
        X[:,8] = X[:,7]
    return N, P, X

def almost_equal(x, x0, threshold=1e-10):
    if x.shape != x0.shape: return False
    
    return np.abs(np.nanmax((x - x0) / x0)) < threshold

def simulate(model, e_data, P, X, plI, plI_int, num_curves,
             sim_params, init_params, sim_flags, gpu_info, gpu_id, solver_time, err_sq_time, misc_time,
             logger=None):
    """ Delegate blocks of simulation tasks to connected GPUs """
    has_GPU = gpu_info["has_GPU"]
    GPU_GROUP_SIZE = gpu_info["sims_per_gpu"]
    num_gpus = gpu_info["num_gpus"]
    
    if has_GPU:
        TPB = gpu_info["threads_per_block"]
        max_sims_per_block = gpu_info["max_sims_per_block"]
        
        try:
            cuda.select_device(gpu_id)
        except IndexError:
            if logger is not None:
                logger.error("Error: threads failed to launch")
            return
        device = cuda.get_current_device()
        num_SMs = getattr(device, "MULTIPROCESSOR_COUNT")
    
    LOADIN_PL = sim_flags["load_PL_from_file"]
    LOG_PL = sim_flags["log_pl"]
    NORMALIZE = sim_flags["self_normalize"]
    

    if isinstance(sim_params[0], (int, float)):
        thicknesses = [sim_params[0] for ic_num in range(num_curves)]
    elif isinstance(sim_params[0], list):
        thicknesses = list(sim_params[0])
        
    # The actual time grid the model will use
    simulation_times = np.linspace(0, sim_params[1], sim_params[3]+1)

    for ic_num in range(num_curves):
        # Update thickness
        sim_params[0] = thicknesses[ic_num]
        if gpu_id == 0 and logger is not None: 
            logger.info("new thickness: {}".format(sim_params[0]))

        sim_params[5] = tuple(np.array(sim_params[5])*sim_params[4]*sim_params[3]//100)

        if gpu_id == 0 and logger is not None: 
            logger.info("Taking {} timesteps".format(sim_params[3]))
            logger.info("Final time: {}".format(sim_params[1]))

            logger.info(sim_params)

        for blk in range(gpu_id*GPU_GROUP_SIZE,len(X),num_gpus*GPU_GROUP_SIZE):
            if logger is not None:
                logger.info("Curve #{}: Calculating {} of {}".format(ic_num, blk, len(X)))
            size = min(GPU_GROUP_SIZE, len(X) - blk)

            if not LOADIN_PL:
                plI[gpu_id] = np.empty((size, sim_params[3]+1), dtype=np.float32) #f32 or f64 doesn't matter much here
                #assert len(plI[gpu_id][0]) == len(values), "Error: plI size mismatch"

                if has_GPU:
                    plN = np.empty((size, 2, sim_params[2]))
                    plP = np.empty((size, 2, sim_params[2]))
                    plE = np.empty((size, 2, sim_params[2]+1))
                    solver_time[gpu_id] += model(plI[gpu_id], plN, plP, plE, X[blk:blk+size, :-1], 
                                                 sim_params, init_params[ic_num],
                                                 TPB,8*num_SMs, max_sims_per_block, init_mode="points")
                else:
                    solver_time[gpu_id] += model(plI[gpu_id], X[blk:blk+size], sim_params, init_params[ic_num])
        
                if NORMALIZE:
                    # Normalize each model to its own t=0
                    plI[gpu_id] = plI[gpu_id].T
                    plI[gpu_id] /= plI[gpu_id][0]
                    plI[gpu_id] = plI[gpu_id].T
                if LOG_PL:
                    if has_GPU:
                        misc_time[gpu_id] += fastlog(plI[gpu_id], sys.float_info.min, TPB[0], num_SMs)
                    else:
                        clock0 = time.perf_counter()
                        plI[gpu_id] = np.abs(plI[gpu_id] + sys.float_info.min)
                        plI[gpu_id] = np.log10(plI[gpu_id])
                        misc_time[gpu_id] += time.perf_counter() - clock0
            else:
                raise NotImplementedError("load PL not implemented")
                #print("Loading plI group {}".format(blk))
                #plI = load_raw_pl(out_filename, ic_num, blk)
                
            for e, exp in enumerate(e_data):
                times = exp[0][ic_num]
                values = exp[1][ic_num]
                std = exp[2][ic_num]
                
                skip_time_interpolation = almost_equal(simulation_times, times)
                
                if logger is not None:
                    if skip_time_interpolation:    
                        logger.info("Experiment {}: No time interpolation needed; bypassing".format(e))
                    else:
                        logger.info("Experiment {}: time interpolating".format(e))
                    
                # Interpolate if observation times do not match simulation time grid
                if skip_time_interpolation:
                    plI_int[gpu_id] = plI[gpu_id]
                else:
                    clock0 = time.perf_counter()
                    plI_int[gpu_id] = np.empty((size, len(times)))
                    
                    for i, unint_PL in enumerate(plI[gpu_id]):
                        plI_int[gpu_id][i] = griddata(simulation_times, unint_PL, times)
                        
                    misc_time[gpu_id] += time.perf_counter() - clock0
                    
                # Calculate errors
                if has_GPU:
                    err_sq_time[gpu_id] += prob(P[e, blk:blk+size], plI_int[gpu_id], values, std, np.ascontiguousarray(X[blk:blk+size, -1]), 
                                                TPB[0], num_SMs)
                    
                else:
                    clock0 = time.perf_counter()
                    P[e, blk:blk+size] -= np.sum((plI_int[gpu_id] - values)**2, axis=1)
                    err_sq_time[gpu_id] += time.perf_counter() - clock0
        # END LOOP OVER BLOCKS
    # END LOOP OVER ICs

    return

def bayes(model, N, P, minX, maxX, do_log, init_params, sim_params, e_data, sim_flags, gpu_info, logger=None):
    """ "main" driver function """
    num_gpus = gpu_info["num_gpus"]
    solver_time = np.zeros(num_gpus)
    err_sq_time = np.zeros(num_gpus)
    misc_time = np.zeros(num_gpus)

    num_curves = len(init_params)
    #timepoints_per_ic = sim_params[3] // sim_params[4] + 1
    #for i in range(len(e_data[0])):
    #    assert (len(e_data[0][i]) % timepoints_per_ic == 0), "Error: experiment {} data length not a multiple of points_per_ic".format(i+1)

    N, P, X = make_grid(N, P, len(e_data), minX, maxX, do_log, sim_flags)
    
    if logger is not None:
        logger.info("Initializing {} random samples".format(len(X)))

    sim_params = [list(sim_params) for i in range(num_gpus)]

    threads = []
    plI = [None for i in range(num_gpus)]
    plI_int = [None for i in range(num_gpus)]
    
    gpu_id = 0
    simulate(model, e_data, P, X, plI, plI_int,
                                  num_curves,sim_params[gpu_id], init_params, sim_flags, gpu_info, gpu_id,
                                  solver_time, err_sq_time, misc_time, logger=logger)
    # for gpu_id in range(num_gpus):
    #     print("Starting thread {}".format(gpu_id))
    #     thread = threading.Thread(target=simulate, args=(model, e_data, P, X, plI, plI_int,
    #                               num_curves,sim_params[gpu_id], init_params, sim_flags, gpu_info, gpu_id,
    #                               solver_time, err_sq_time, misc_time, logger=logger))
    #     threads.append(thread)
    #     thread.start()

    # for gpu_id, thread in enumerate(threads):
    #     print("Ending thread {}".format(gpu_id))
    #     thread.join()
    #     print("Thread {} closed".format(gpu_id))

    if logger is not None:
        logger.info("Total tEvol time: {}, avg {}".format(solver_time, np.mean(solver_time)))
        logger.info("Total err_sq time (temperatures and mag_offsets): {}, avg {}".format(err_sq_time, np.mean(err_sq_time)))
        logger.info("Total misc time: {}, avg {}".format(misc_time, np.mean(misc_time)))
    return N, P, X


#-----------------------------------------------------------------------------#