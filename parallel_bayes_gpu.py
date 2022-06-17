"""
Last edited on Tue Feb 26 11:54:38 2022

@author: cfai2304

Main entry point for inferencing algorithm. Run this as main.
"""
import logging
from numba import cuda
from time import perf_counter
import sys
import numpy as np
import os

from bayeslib import bayes
from bayes_io import get_initpoints, get_data, export
from bayes_validate import validate_ic_flags, validate_gpu_info
from bayes_validate import validate_IC, validate_params
from bayes_validate import connect_to_gpu
from pvSimPCR import pvSim

lambda0 = 704.3                           # q^2/(eps0*k_B T=25C) [nm]
param_names = ["n0", "p0", "mun", "mup", "B", "Sf", "Sb", "taun", "taup", "lambda", "mag_offset"]

# Convert from common units to [V, nm, ns]
unit_conversions = np.array([(1e7)**-3,(1e7)**-3,
                             (1e7)**2/(1e9)*.02569257,(1e7)**2/(1e9)*.02569257,
                             (1e7)**3/(1e9),
                             (1e7)/(1e9),(1e7)/(1e9),
                             1,1,
                             lambda0, 1])
num_params = len(param_names)
np.random.seed(42)
if __name__ == "__main__":

    # Set space and time grid options
    #Length = [311,2000,311,2000, 311, 2000]
    Length  = 311                            # Length (nm)
    L   = 2 ** 7                                # Spatial points
    plT = 1                                  # Set PL interval (dt)
    pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
    tol = 7                                   # Convergence tolerance
    MAX = 10000                                  # Max iterations

    simPar = [Length, -1, L, -1, plT, pT, tol, MAX]

    # This code follows a strict order of parameters:
    # matPar = [N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda, mag_offset]
    # Set the parameter ranges/sample space
    do_log = np.array([1,1,0,0,1,1,1,0,0,1,0])
    minX = np.array([1e8, 1e15, 0, 0, 1e-11, 0.1, 0.1, 1, 1, 10**-1, 0])
    maxX = np.array([1e8, 1e16, 50, 50, 1e-9, 1e2, 1e2, 1000, 2000, 10**-1, 0])

    # Other options
    # time_cutoff: Truncate observations with timestamps larger than time_cutoff.
    # select_obs_sets: Drop selected observation sets. [0,2] drops the 1st and 3rd observation sets.
    # noise_level: Add Gaussian noise with this sigma to observation sets.
    ic_flags = {"time_cutoff":None,
                "select_obs_sets":None,
                "noise_level":None}

    # sims_per_gpu: Number of simulations dispatched to GPU at a time. Adjust according to GPU mem limits.
    # num_gpus: Number of GPUs to attempt connecting to.
    gpu_info = {"sims_per_gpu": 2 ** 10,    
                "num_gpus": 1}


    # load_PL_from_file: Whether to import TRPL simulation data. Currently doesn't do anything.
    # override_equal_mu: Constrain sampled mu_n to equal sampled mu_p.
    # override_equal_s: Constrain sampled Sb to equal Sf.
    # "log_pl: Compare log10 of PL for likelihood rather than direct PL values.
    # self_normalize: Normalize all observed and simulation TRPL curves to their own maxima.init_dir
    # random_sample: Draw random samples from uniform parameter space.
    # num_points: Number of random samples to draw.
    sim_flags = {"load_PL_from_file": False,
                 "override_equal_mu":False,
                 "override_equal_s":False,
                 "log_pl":True,
                 "self_normalize":False,
                 "random_sample":True,
                 "num_points":2**17, 
                 "different_time_grid":None}#(2000, 0.025)}

    # Collect filenames
    init_dir = r"C:\Users\Chuck\Dropbox (UFL)\UF\Bayesian-Inference-TRPL Data\Staubb_Simulated\bay_inputs"
    out_dir = r"C:\Users\Chuck\Dropbox (UFL)\UF\Bayesian-Inference-TRPL Data\Staubb_Simulated\bay_outputs"
    init_filename = os.path.join(init_dir, "staub_MAPI_power_input.csv")
    experimental_data_filename = os.path.join(init_dir, "staub_311nm_minsf.csv")
    out_filename = os.path.join(out_dir, "OUT_staub_MAPI_power")

    # Get observations and initial condition
    iniPar = get_initpoints(init_filename, ic_flags)
    e_data = get_data(experimental_data_filename, ic_flags, sim_flags, scale_f=1e-23)

    # Validate
    try:
        assert len(iniPar) == len(e_data[0]), "Num. ICs mismatch num. datasets"
        validate_ic_flags(ic_flags)
        validate_IC(iniPar, L)
        validate_gpu_info(gpu_info)
        validate_params(num_params, unit_conversions, do_log, minX, maxX)
        connect_to_gpu(gpu_info, nthreads=128, sims_per_block=1)
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    print("Starting simulations with the following parameters:")

    print("Lengths: {}".format(Length))
    for i in range(num_params):
        if minX[i] == maxX[i]:
            print("{}: {}".format(param_names[i], minX[i]))

        else:
            print("{}: {} to {} {}".format(param_names[i], minX[i], maxX[i], "log" if do_log[i] else "linear"))

    print("\nInitial condition - {}".format(init_filename))
    print("\nExperimental data - {}".format(experimental_data_filename))
    print("Output: {}".format(out_filename))

    print(ic_flags)
    print(gpu_info)
    print(sim_flags)

    minX *= unit_conversions
    maxX *= unit_conversions

    N    = np.array([0])
    P    = None
    clock0 = perf_counter()
    N, P, X = bayes(pvSim, N, P, minX, maxX, do_log, iniPar, simPar, e_data, sim_flags, gpu_info)
    print("Bayesim took {} s".format(perf_counter() - clock0))

    minX /= unit_conversions
    maxX /= unit_conversions
    X /= unit_conversions

    clock0 = perf_counter()
    export(out_filename, P, X)
    print("Export took {}".format(perf_counter() - clock0))
