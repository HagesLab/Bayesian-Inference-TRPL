# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 21:04:17 2020

@author: cfai2
(TESTING) Generate a set of simulations from scipy odeint/solve_ivp
"""
import pickle
import numpy as np
from scipy.integrate import odeint
from numba import njit

@njit(cache=True)
def dydt(t, y, L, N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda):
    Jn = np.zeros((L+1))
    Jp = np.zeros((L+1))
    dydt = np.zeros(3*L+1)

    N = y[0:L]
    P = y[L:2*(L)]
    E_field = y[2*(L):]
    
    NP = (N * P - N0 * P0)
    
    Sft = sr0 * NP[0] / (N[0] + P[0])
    Sbt = srL * NP[-1] / (N[-1] + P[-1])
    

    Jn[0] = Sft
    Jn[L] = -Sbt
    Jp[0] = -Sft
    Jp[L] = Sbt

    # DN = mu_n*kB*T/q
    for i in range(1, len(Jn) - 1):
        Jn[i] = DN*((N[i-1] + N[i]) / 2 * E_field[i] + (N[i] - N[i-1]))
        Jp[i] = DP*((P[i-1] + P[i]) / 2 * E_field[i] - (P[i] - P[i-1]))
    
    for i in range(len(Jn)):
        dydt[2*L+i] = -(Jn[i] + Jp[i]) * Lambda
    
    recomb = (rate + 1 / ((tauN * P) + (tauP * N))) * NP

    
    for i in range(len(Jn) - 1):
        dydt[i] = (Jn[i+1] - Jn[i] - recomb[i])
        dydt[L+i] = (-Jp[i+1] + Jp[i] - recomb[i])

    return dydt


if __name__ == "__main__":
    simPar, iniPar, matPar = pickle.load(open('hagesInputs302.pik', 'rb'))
    # Unpack local parameters
    Length, Time, L, T, plT, pT, TOL, MAX = simPar
    dx = Length/L
    dt = Time/T
    
    # Non dimensionalize variables
    dx3 = dx**3; dtdx = dt/dx; dtdx2 = dtdx/dx
    matPar  = np.array(matPar)
    scales  = np.array([dx3,dx3,dtdx2,dtdx2,dtdx2/dx,dtdx,dtdx,1/dt,1/dt,1/dx])
    matPar *= scales
    
    a, l = iniPar
    a  *= dx3
    l  /= dx
    N0   = matPar[:,0]
    P0   = matPar[:,1]
    rate = matPar[:,4]
    x  = np.arange(L) + 0.5
    dN = a * np.exp(-x/l)
    
    nthr = len(matPar)
    plN = np.zeros((nthr, len(pT), L))
    plP = np.zeros((nthr, len(pT), L))
    plE = np.zeros((nthr, len(pT), L+1))
    plI = np.zeros((nthr, T//plT+1))
    
    import time
    startTime = time.time()
    haches = np.zeros(nthr)
    methods = np.zeros((nthr, T), dtype=int)
    orders = np.zeros((nthr, T), dtype=int)
    Threads = [69, 420, 1420, 2580]
    
    #for thr in Threads:
    for thr in range(nthr):
        print(thr)
        init_N = N0[thr] + dN
        init_P = P0[thr] + dN
        init_E = np.zeros(L+1)
        
        init_condition = np.concatenate([init_N, init_P, init_E], axis=None)
    
        tSteps = np.linspace(0, T, T+1)
        data, error_dict = odeint(dydt, init_condition, tSteps, args=(L, *matPar[thr]),\
            tfirst=True, full_output=True)
            
        N = data[:,0:L]
        P = data[:,L:2*(L)]
        E = data[:,2*(L):]
        h = 4.0
        while True:
            if (N < 0).any() or (P < 0).any():
                print("h=",h)
                haches[thr] = h
                data, error_dict = odeint(dydt, init_condition, tSteps, args=(L, *matPar[thr]),\
                              tfirst=True, hmax=h, full_output=True)
                h /= 2
                
                N = data[:,0:L]
                P = data[:,L:2*(L)]
                E = data[:,2*(L):]
                

            else:
                break
        
        plI[thr] = rate[thr] * np.sum(N[::plT] * P[::plT] - N0[thr] * P0[thr], axis=1) 
        for t in pT:
            plN[thr, pT.index(t)] = N[t]
            plP[thr, pT.index(t)] = P[t]
            plE[thr, pT.index(t)] = E[t]
            
        methods[thr] = error_dict['mused']
        orders[thr] = error_dict['nqu']
            
    print("Took {} sec".format(time.time() - startTime))
    plI *= dx**4/dt
    plN /= dx**3
    plP /= dx**3
    plE /= dx
    
    pickle.dump((plN, plP, plE, plI), open('testHagesOut300.pik', 'wb'))