# Bayesian-Inference-TRPL
A Bayesian inference algorithm and script for the fitting of TRPL observations from absorbers to the carrier drift/diffusion/decay model described in *Minority Carriers in III-V Semiconductors: Physics and Applications* (Ahrenkiel and Lundstrom, 1993) and *Chalcogenide Photovoltaics: Physics, Technologies, and Thin Film Devices* (Scheer and Schock, 2011). See our *Adv. Energy Mater.* article\[under review\] for additional details.

## File Overview
* parallel_bayes_gpu.py - Entry script and configuration settings for the inference. Do `python parallel_bayes_gpu.py OBSERVATION_FILE EXCITATION_FILE OUTPUT_NAME [new|new+|load]` to start the show. 
* bayeslib.py - Driver functions for generating random samples of the parameter space, delegating simulations to GPUs, and computing resultant likelihoods.
* pvSimPCR.py - GPU solver for TRPL simulation. Uses parallel cyclic reduction. Can be run directly as a standalone simulator when provided with an input pickle.
* probs.py - GPU-accelerated likelihood computations.
* bayes_io.py - Utility functions for loading OBSERVATION and EXCITATION files and exporting output likelihood files.
* bayes_validate.py - Validation for configuration settings in *parallel_bayes_gpu.py*.
* **Example Data**
  * Power_scan_Excitations.csv - Excitations/Initial Conditions for the "exemplary" three-level power scan shown in the article.
  * Power_scan_Observations.csv - Observations/TRPL data points for the "exemplary" three-level power scan shown in the article. Goes with Power_scan_Excitations.csv.
  * Highsurf_Power_scan_Observations - Observations/TRPL data points for the "alternate high surface" power scan shown in the article. Also uses Power_scan_Excitations.csv.
  * Twothick_Excitations.csv - Excitations/Initial Conditions for the two-thickness scan discussed in the article.
  * Twothick_Observations.csv - Observations/TRPL data points for the two-thickness scan discussed in the article. Goes with Twothick_Excitations.csv
* **Testing**
  * pvSetup.py - Generate an input pickle for running pvSimPCR.py standalone.
  * PV_tester2.py - Simulate TRPL using input pickle and scipy PDE solvers. Generates output pickle containing simulation results.
  * compare.py - Compare two output pickles. Perhaps one from pvSimPCR.py and one from PV_tester2.py?
  * pvPlt_interface.py - Basic visualizer for output pickles.
* **Legacy**
  * legacy.py - Functions for coarse grid sampling of the parameter space.
  * parallel_bayes.py - Serial CPU version of the Bayesian inference algorithm.
  * pvSim.py - Serial CPU version of the TRPL simulation solver.

## Usage
1. Configure parallel_bayes_gpu.py. Adjust simPar, parameter space boundaries, ic_flags, gpu_info, sim_flags, init_dir, and out_dir.
2. Prepare OBSERVATION and EXCITATION files. The format of these are as follows: OBSERVATION files are three column lists of TRPL data points: timestamp, TRPL value, and experimental uncertainty value. If multiple TRPL curves are to be inferenced, the sets of data points for each curve should be concatenated one below the next. Add a single line with the word "END" to the end of the list. EXCITATION files are lists of initial carrier density profiles - each row corresponding to one curve in the OBSERVATION file. See **Example Data** for exmaples.
3. Run `python parallel_bayes_gpu.py OBSERVATION_FILE EXCITATION_FILE OUTPUT_NAME [new|new+|load]`.
4. Collect "BAYRAN_X" parameter sample file and "BAYRAN_P" raw likelihood file.
5. Compute posterior probability distribution from "BAYRAN_X" and "BAYRAN_P". Example script coming soon!
