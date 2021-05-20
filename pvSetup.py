#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:29:39 2020

@author: tladd
"""
def get_all_combinations(value_array):
    num_params = len(value_array)
    n = 1
    cn = 0
    iterable_param_indexes = [0] * num_params
    iterable_param_lengths = [1] * num_params
    for val in range(num_params):
        if isinstance(value_array[val], (float, int)):
            value_array[val] = np.array([value_array[val]])

        n *= len(value_array[val])
        iterable_param_lengths[val] = len(value_array[val])

    combinations = [None] * n

    pivot_index = num_params - 1

    current_params = [0] * num_params
    for i in range(num_params):
        current_params[i] = value_array[i][0]

    while(pivot_index >= 0):

        for i in range(pivot_index, num_params):
            current_params[i] = value_array[i][iterable_param_indexes[i]]

        combinations[cn] = list(current_params)

        pivot_index = num_params - 1
        while (pivot_index >= 0 and iterable_param_indexes[pivot_index] == iterable_param_lengths[pivot_index] - 1):
            pivot_index -= 1

        iterable_param_indexes[pivot_index] += 1

        for i in range(pivot_index + 1, num_params):
            iterable_param_indexes[i] = 0
        cn += 1

    return combinations

if __name__ == "__main__":

    import pickle
    import numpy as np
    dir = r"/blue/c.hages/cfai2304/"
    fname = 'ipvtest.pik'
    # simPar
    Time    = 100                             # Final time (ns)
    Length  = 1000                            # Length (nm)
    lambda0 = 704.3                           # q^2/(eps0*k_B T=25C) [nm]
    L   = 2 ** 7                                # Spatial points
    T   = 4000                                # Time points
    plT = 10                                  # Set PL interval (dt)
    pT  = (0,1,3,10,30,100)                   # Set plot intervals (%)
    tol = 5                                   # Convergence tolerance
    MAX = 500                                  # Max iterations

    # iniPar
    a  = 1e18/(1e7)**3                        # Amplitude
    l  = 100                                  # Length scale [nm]
    N0 = 1e8 /(1e7)**3                        # [/ nm^3]
    P0 = 1e16/(1e7)**3                        # [/ nm^3]

    # matPar - 1 cm^2/Vs = 2.569257e3 nm^2/ns
    DN  = np.array([0.1,10])  *(1e7)**2/(1e9)*.02569257      # [nm^2 / ns]
    DP  = np.array([0.1,10])  *(1e7)**2/(1e9)*.02569257      # [nm^2  /ns]
    rate = np.array([1e-10,1e-12])   *(1e7)**3/(1e9)            # [nm^3 / ns]
    sr0  = np.array([1e2, 1e5]) *(1e7)/(1e9)               # [nm / ns]
    srL  = np.array([1e2, 1e5]) *(1e7)/(1e9)               # [nm / ns]
    tauN = np.array([.5,50])                                 # [ns]
    tauP = np.array([.5,50])                                 # [ns]
    Lambda = lambda0*np.array([10.])**-1                   # [nm]

    # Pack parameters
    pT = tuple(np.array(pT)*T//100)
    simPar = (Length, Time, L, T, plT, pT, tol, MAX)
    iniPar = [[a, l]]
    matPar = [N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda]
    matPar = get_all_combinations(matPar)

    pickle.dump((simPar,iniPar,matPar), open(dir + fname, 'wb'))
    print("wrote input pickle ", dir + fname)
