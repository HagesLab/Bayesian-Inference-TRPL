import csv
import logging
from numba import cuda
from time import perf_counter
import sys
import numpy as np

from bayeslib import bayes, get_initpoints, get_data, validate_ic_flags, validate_gpu_info, validate_IC
from pvSimPCR import pvSim

lambda0 = 704.3                           # q^2/(eps0*k_B T=25C) [nm]
param_names = ["n0", "p0", "mun", "mup", "B", "Sf", "Sb", "taun", "taup", "lambda", "mag_offset"]
unit_conversions = np.array([(1e7)**-3,(1e7)**-3,
                             (1e7)**2/(1e9)*.02569257,(1e7)**2/(1e9)*.02569257,
                             (1e7)**3/(1e9),
                             (1e7)/(1e9),(1e7)/(1e9),
                             1,1,
                             lambda0, 1])

np.random.seed(42)
if __name__ == "__main__":

    ic_flags = {"time_cutoff":None,
                "select_obs_sets":None,
                "noise_level":1e15}
    # simPar
    #Length = [311,2000,311,2000, 311, 2000]
    Length  = 2000                            # Length (nm)
    L   = 2 ** 7                                # Spatial points
    plT = 1                                  # Set PL interval (dt)
    pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
    tol = 7                                   # Convergence tolerance
    MAX = 10000                                  # Max iterations

    simPar = [Length, -1, L, -1, plT, pT, tol, MAX]

    # iniPar and available modes
    # 'exp' - parameters a and l for a*np.exp(-x/l)
    # 'points' - direct list of dN [cm^-3] points as csv file read using get_initpoints()
    #TODO: do away with init_mode
    init_mode = "points"
    #a  = 1e18/(1e7)**3                        # Amplitude
    #l  = 100                                  # Length scale [nm]
    #iniPar = np.array([[a, l]])
    
    try:
        validate_ic_flags(ic_flags)
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    iniPar = get_initpoints(r"/blue/c.hages/bay_inputs/{}".format(sys.argv[2]), ic_flags)
    try:
        validate_IC(iniPar, L)
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    # This code follows a strict order of parameters:
    # matPar = [N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda, mag_offset]
    # Parameter ranges
    do_log = np.array([1,1,0,0,1,1,1,0,0,1,0])
    minX = np.array([1e8, 3e15, 4000, 4000, 4.8e-11, 1, 1, 1, 1, 10**-1, 0])
    maxX = np.array([1e8, 3e15, 4000, 4000, 4.8e-11, 1e5, 1e5, 1500, 3000, 10**-1, 0])

    GPU_GROUP_SIZE = 2 ** 13                  # Number of simulations assigned to GPU at a time - GPU has limited mem$
    num_gpus = 8
    
    gpu_info = {"sims_per_gpu": 2 ** 13,
                "num_gpus": 8}

    try:
        validate_gpu_info(gpu_info)
    except Exception as e:
        logging.error(e)
        sys.exit(1)
        
    sim_flags = {"load_PL_from_file": "load" in sys.argv[4],
                 "override_equal_mu":False,
                 "override_equal_s":False,
                 "log_pl":True,
                 "self_normalize":False,
                 "random_sample":True,
                 "num_points":2**17}

    experimental_data_filename = r"/blue/c.hages/bay_inputs/{}".format(sys.argv[1])
    out_filename = sys.argv[3]
    wdir = r"/blue/c.hages/bay_outputs/{}/".format(out_filename)

    # Pre-checks
    try:
        num_params = len(param_names)
        assert (len(unit_conversions) == num_params), "Unit conversion array is missing entries"
        assert (len(do_log) == num_params), "do_log mask is missing values"
        assert (len(minX) == num_params), "Missing min param values"
        assert (len(maxX) == num_params), "Missing max param values"
        assert all(minX <= maxX), "Min params larger than max params"
        

        print("Starting simulations with the following parameters:")

        print("Lengths: {}".format(Length))
        for i in range(num_params):
            if minX[i] == maxX[i]:
                print("{}: {}".format(param_names[i], minX[i]))

            else:
                print("{}: {} to {} {}".format(param_names[i], minX[i], maxX[i], "log" if do_log[i] else "linear"))

        e_data = get_data(experimental_data_filename, ic_flags, sim_flags, scale_f=1e-23)
        assert len(iniPar) == len(e_data[0]), "Num. ICs mismatch num. datasets"
        print("\nInitial condition - {}".format(iniPar))
        print("\nExperimental data - {}".format(experimental_data_filename))

        print(e_data)
        print("Output: {}".format(out_filename))
        try:
            print("Detecting GPU...")
            gpu_info["has_GPU"] = cuda.detect()
        except Exception as e:
            print(e)
            gpu_info["has_GPU"] = False

        if gpu_info["has_GPU"]:
            device = cuda.get_current_device()

            gpu_info["threads_per_block"] = (2 ** 7,)
            gpu_info["max_sims_per_block"] = 3           # Maximum of 6 due to shared memory limit
            
        else:
            print("No GPU detected - reverting to CPU simulation")
            raise NotImplementedError
            

    except Exception as oops:
        print(oops)
        sys.exit(0)

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

    try:
        import os
        print("Creating dir {}".format(out_filename))
        os.mkdir(wdir)
    except FileExistsError:
        print("{} dir already exists".format(out_filename))
    clock0 = perf_counter()

    try:
        print("Writing to /blue:")
        np.save("{}_BAYRAN_P.npy".format(wdir + out_filename), P)
        np.save("{}_BAYRAN_X.npy".format(wdir + out_filename), X)

    except Exception as e:
        print(e)
        print("Write failed; rewriting to backup location /home:")
        wdir = r"/home/cfai2304/super_bayes/"
        np.save("{}_BAYRAN_P.npy".format(wdir + out_filename), P)
        np.save("{}_BAYRAN_X.npy".format(wdir + out_filename), X)

    print("Export took {}".format(perf_counter() - clock0))


