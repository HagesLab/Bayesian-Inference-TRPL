# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:59:54 2020

@author: cfai2
(TESTING) Script for benchmarking GPU solver against scipy.solve_ivp() reference
"""

import pickle
import numpy as np
import scipy.linalg as lin
names = ("N", "P", "E", "PL")

# Enter two sets of data and input pickles
actuals = pickle.load(open('hagesOut300.pik', 'rb'))
reference = pickle.load(open('testHagesOut300.pik', 'rb'))
Length, Time, L, T, plT, pT, TOL, MAX = pickle.load(open('hagesInputs300.pik', 'rb'))[0]
rLength, rTime, rL, rT, rplT, rpT, rTOL, rMAX = pickle.load(open('hagesInputs300.pik', 'rb'))[0]

nthr = len(actuals[0])
# Space sample for N, P, E norms
test_locs = np.array([0.1*L, 0.3*L, 0.5*L, 0.7*L, 0.9*L], dtype=int)
rtest_locs = np.array([0.1*rL, 0.3*rL, 0.5*rL, 0.7*rL, 0.9*rL], dtype=int)


plI = actuals[-1]
test_plI = reference[-1]
m = len(plI[0])
rm = len(test_plI[0])
# Time sample for PL norm
test_times = np.array([0*m, 0.01*m, 0.03*m, 0.1*m, 0.3*m, m-1], dtype=int)
rtest_times = np.array([0*rm, 0.01*rm, 0.03*rm, 0.1*rm, 0.3*rm, rm-1], dtype=int)
figc = 0
nerr = np.zeros(len(names))

for thr in range(nthr):

    for test in range(len(actuals) - 1):# For N, P, E
        
        try:
            ndiff = lin.norm(actuals[test][thr,:,test_locs].flatten() - reference[test][thr,:,rtest_locs].flatten()) / lin.norm(reference[test][thr,:,rtest_locs].flatten())
            nerr[test] += ndiff
            if ndiff > 10:
                print("Warning: Thread {} had ndiff={} for {}".format(thr, ndiff, names[test]))
        except:
            nerr[test] += 1
            print("Error: invalid value in ", names[test], " for Thread ", thr)


    try:
        nerr[-1] += lin.norm(plI[thr, test_times] - test_plI[thr, rtest_times]) / lin.norm(test_plI[thr, rtest_times])   
    except:
        nerr[-1] += 1
        print("Error: invalid value in PL for Thread ", thr)
            
        
nerr /= nthr
for name in range(len(names)):
    print("Average norm_error ", names[name], ': ', nerr[name])