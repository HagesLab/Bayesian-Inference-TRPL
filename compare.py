# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:59:54 2020

@author: cfai2
"""

import pickle
import numpy as np
import scipy.linalg as lin
actuals = pickle.load(open('hagesOut.pik', 'rb'))
names = ("N", "P", "E", "PL")
reference = pickle.load(open('testHagesOut.pik', 'rb'))
Length, Time, L, T, plT, pT, TOL, MAX = pickle.load(open('pvInputs.pik', 'rb'))[0]

nthr = len(actuals[0])
# Space sample for N, P, E norms
test_locs = np.array([0.1*L, 0.3*L, 0.5*L, 0.7*L, 0.9*L], dtype=int)


plI = actuals[-1]
test_plI = reference[-1]
m = len(plI[0])
# Time sample for PL norm
test_times = np.array([0*m, 0.01*m, 0.03*m, 0.1*m, 0.3*m, m-1], dtype=int)
figc = 0
nerr = np.zeros(len(names))

for thr in range(nthr):

    for test in range(len(actuals) - 1):# For N, P, E
        
        try:
            nerr[test] += lin.norm(actuals[test][thr,:,test_locs].flatten() - reference[test][thr,:,test_locs].flatten()) / lin.norm(reference[test][thr,:,test_locs].flatten())
        except:
            nerr[test] += 1
            print("Error: invalid value in ", names[test], " for Thread ", thr)


    try:
        nerr[-1] += lin.norm(plI[thr, test_times] - test_plI[thr, test_times]) / lin.norm(test_plI[thr, test_times])   
    except:
        nerr[-1] += 1
        print("Error: invalid value in ", names[test], " for Thread ", thr)
            
        
nerr /= nthr
for name in range(len(names)):
    print("Average error ", names[name], ': ', nerr[name])