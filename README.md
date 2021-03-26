# Super_bayes
An adaptation of the bayesian inference procedure for a parallel processing engine

## Files
* ...bayesim example.h5 - sample experimental data for bayesian inference. This will be deprecated when parallel_bayes.py is adapted to work with pvSim's output pickles
* parallel_bayes.py - bayesian inferencing script for nano simulation. Uses output pickles + experimental data to fit parameters.
* testmodel_bayes.py - test bayesian inferencing script that fits a simple stiff ODE.
* PV_tester2.py - ODEint solver that produces an output pickle similar to pvSim's. Used for benchmarking pvSim.
* ... true param vals.txt - the actual parameters from the sample experimental data. Used to benchmark parallel_bayes.py
* compare.py - Benchmarks pvSim.py using output pickles.
* pvPlt.py - Plots output pickle data for selected parameter set together for visual comparison. Interface version has a GUI for changing between parameter sets.
* pvSetup.py - Generates pvSim input pickles.
* i600.pik - Example pvSim input pickle.
* pvSim.py - CPU solver
* pvSimPCR.py - GPU solver. Uses parallel cyclic reduction.
