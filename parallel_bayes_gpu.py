"""
Last edited on Tue Feb 26 11:54:38 2022

@author: cfai2304

Main entry point for inferencing algorithm. Run this as main.
"""
import logging
from numba import cuda
from time import perf_counter
from datetime import datetime
import sys
import numpy as np
import os

from bayeslib import bayes
from bayes_io import get_initpoints, get_data, export
from bayes_validate import validate_ic_flags, validate_gpu_info
from bayes_validate import validate_IC, validate_params
from bayes_validate import connect_to_gpu


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

def start_logging(log_dir="Logs"):
    if not os.path.isdir(log_dir):
        try:
            os.mkdir(log_dir)
        except FileExistsError:
            pass
        
    tstamp = str(datetime.now()).replace(":", "-")
    #logging.basicConfig(filename=os.path.join(log_dir, f"{tstamp}.log"), filemode='a', level=logging.DEBUG)
    logger = logging.getLogger("Bayes Logger Main")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(os.path.join(log_dir, f"{tstamp}.log"))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
            )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger, handler

def stop(logger, handler, err=0):
    if err:
        logger.error(f"Termining with error code {err}")
    # Spyder needs explicit handler handling for some reason
    logger.removeHandler(handler)
    logging.shutdown()
    return
    
if __name__ == "__main__":
    logger, handler = start_logging()

    # Set space and time grid options for simulations
    #Length = [311,2000,311,2000, 311, 2000]
    Length  = 311                            # Length (nm)
    L   = 2 ** 7                                # Spatial points
    Time = 2000                             # Final delay time (ns)
    T = 80000                               # Time points
    plT = 1                                  # Set PL interval (dt)
    pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
    tol = 7                                   # Convergence tolerance
    MAX = 10000                                  # Max iterations

    simPar = [Length, Time, L, T, plT, pT, tol, MAX]

    # This code follows a strict order of parameters:
    # matPar = [N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda, mag_offset]
    # Set the parameter ranges/sample space
    do_log = np.array([1,1,0,0,1,1,1,0,0,1,0])
    minX = np.array([1e8, 1e14, 0, 0, 1e-11, 0.1, 0.1, 1, 1, 10**-1, 0])
    maxX = np.array([1e8, 1e16, 50, 50, 1e-9, 1e2, 1e2, 1000, 2000, 10**-1, 0])

    # Other options
    # time_cutoff: Truncate observations with timestamps larger than time_cutoff.
    # select_obs_sets: Drop selected observation sets. [0,2] drops the 1st and 3rd observation sets.
    # noise_level: Add Gaussian noise with this sigma to observation sets.
    ic_flags = {"time_cutoff":2000,
                "select_obs_sets":None,
                "noise_level":None}

    # sims_per_gpu: Number of simulations dispatched to GPU at a time. Adjust according to GPU mem limits.
    # num_gpus: Number of GPUs to attempt connecting to.
    gpu_info = {"sims_per_gpu": 2 ** 2,    
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
                 "num_points":2**3, 
                 }

    # Collect filenames
    init_dir = r"C:\Users\cfai2\Documents\src\bayesian processing\input curves\Staubb_Simulated\bay_inputs"
    out_dir = r"C:\Users\cfai2\Documents\src\bayesian processing\input curves\Staubb_Simulated\bay_outputs"
    init_filename = os.path.join(init_dir, "staub_MAPI_power_input.csv")
    experimental_data_filename = [os.path.join(init_dir, "staub_311nm_minsf.csv"),
                                  os.path.join(init_dir, "staub_311nm_minsf_staubtimeres.csv"),]
    out_filename = [os.path.join(out_dir, "TEST0"),
                    os.path.join(out_dir, "TEST1"),]

    # Get observations and initial condition
    iniPar = get_initpoints(init_filename, ic_flags)
    e_data = get_data(experimental_data_filename, ic_flags, sim_flags, logger=logger, scale_f=1e-23)

    # Validate
    try:
        for exp in e_data:
            assert len(iniPar) == len(exp[0]), "Num. ICs mismatch num. datasets"
        validate_ic_flags(ic_flags)
        validate_IC(iniPar, L)
        validate_gpu_info(gpu_info)
        validate_params(num_params, unit_conversions, do_log, minX, maxX)
    except AssertionError as e:
        logger.error("Validation error")
        logger.error(e)
        stop(logger, handler, 1)
        
    try:
        connect_to_gpu(gpu_info, nthreads=128, sims_per_block=1)
    except Exception as e:
        logging.warning(e)
        logging.warning("Continuing with CPU fallback")

    if gpu_info.get('has_GPU', False):
        from pvSimPCR import pvSim
        model = pvSim
        
    else:
        from pvSim_fallback import pvSim_cpu_fallback
        model = pvSim_cpu_fallback
        
    logger.info("Starting simulations with the following parameters:")

    logger.info("Lengths: {}".format(Length))
    for i in range(num_params):
        if minX[i] == maxX[i]:
            logger.info("{}: {}".format(param_names[i], minX[i]))

        else:
            logger.info("{}: {} to {} {}".format(param_names[i], minX[i], maxX[i], "log" if do_log[i] else "linear"))

    logger.info("\nInitial condition - {}".format(init_filename))
    logger.info("\nExperimental data - {}".format(experimental_data_filename))
    logger.info("Output: {}".format(out_filename))

    logger.info(ic_flags)
    logger.info(gpu_info)
    logger.info(sim_flags)

    minX *= unit_conversions
    maxX *= unit_conversions

    N    = np.array([0])
    P    = None
    clock0 = perf_counter()
    N, P, X = bayes(model, N, P, minX, maxX, do_log, iniPar, simPar, e_data, sim_flags, gpu_info)
    logger.info("Bayesim took {} s".format(perf_counter() - clock0))

    minX /= unit_conversions
    maxX /= unit_conversions
    X /= unit_conversions

    clock0 = perf_counter()
    for i, of in enumerate(out_filename):
        export(of, P[i], X, logger=logger)
    logger.info("Export took {}".format(perf_counter() - clock0))
    stop(logger, handler, 0)