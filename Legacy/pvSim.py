#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:05:30 2020

(DEPRECATED) PV simulator - CPU version - Tony Ladd and Calvin Fai
"""


import numpy as np
from numba import njit
import time

@njit(cache=True)
def Solve(A, b, c):                                 # Solve with norm check

    n = len(b)
    sumx = np.abs(A[1,0]*c[0]+A[0,1]*c[1] - b[0])   # norms
    sumb = np.abs(b[0])
    for k in range(1,n-1):
        sumx += np.abs(A[2,k-1]*c[k-1]+A[1,k]*c[k]+A[0,k+1]*c[k+1] - b[k])
        sumb += np.abs(b[k])
    sumx += np.abs(A[2,-2]*c[-2]+A[1,-1]*c[-1] - b[-1])
    sumb += np.abs(b[-1])

    for k in range(n-1):                            # Elimination
        q = A[2,k]/A[1,k]
        A[1,k+1] -= q*A[0,k+1]
        b[k+1]   -= q*b[k]

    c[-1] = b[-1]/A[1,-1]                           # Backsubstitution
    for k in range(n-2,-1,-1):
        c[k] = (b[k]-A[0,k+1]*c[k+1])/A[1,k]
    return sumx/sumb

@njit(cache=True)
def iterate(N, P, E, A, b, matPar, par):

# Unpack local variables
    N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda = matPar
    a0, a1, a2, k, kp, ko, L, tol , MAX = par
    TOL  = 10.0**(-tol)

# Iterate outer loop to convergence
    for iters in range(MAX):                    # Up to MAX iterations
        for n in range(1,L):                    # Solve for N
            A[0,n]   = DN*(-E[kp,n]/2 - 1)
            A[2,n-1] = DN*(+E[kp,n]/2 - 1)
        for n in range(L):
            np = N[kp,n]*P[kp,n] - N0*P0
            tp = N[kp,n]*tauP + P[kp,n]*tauN
            dR = -rate*P[kp,n] - (P[kp,n]*tp - tauP*np)/tp**2
            A[1,n] = a0 - A[0,n] - A[2,n] - dR
            b[n]   = -(rate + 1/tp)*np - dR*N[kp,n] - a1*N[k,n] - a2*N[ko,n]
        ds0 = -sr0*(P[kp, 0]**2 + N0*P0)/(N[kp, 0]+P[kp, 0])**2
        dsL = -srL*(P[kp,-1]**2 + N0*P0)/(N[kp,-1]+P[kp,-1])**2
        A[1,0]  -= ds0
        A[1,-1] -= dsL
        b[0]  -= sr0*(N[kp, 0]*P[kp, 0]-N0*P0)/(N[kp, 0]+P[kp, 0]) + ds0*N[kp, 0]
        b[-1] -= srL*(N[kp,-1]*P[kp,-1]-N0*P0)/(N[kp,-1]+P[kp,-1]) + dsL*N[kp,-1]
        errN = Solve(A, b, N[kp])

        for n in range(1,L):                    # Solve for P
            A[0,n]   = DP*(+E[kp,n]/2 - 1)
            A[2,n-1] = DP*(-E[kp,n]/2 - 1)
        for n in range(L):
            np = N[kp,n]*P[kp,n] - N0*P0
            tp = N[kp,n]*tauP + P[kp,n]*tauN
            dR = -rate*N[kp,n] - (N[kp,n]*tp - tauN*np)/tp**2
            A[1,n] = a0 - A[0,n] - A[2,n] - dR
            b[n]   = -(rate + 1/tp)*np - dR*P[kp,n] - a1*P[k,n] - a2*P[ko,n]
        ds0 = -sr0*(N[kp, 0]**2 + N0*P0)/(N[kp, 0]+P[kp, 0])**2
        dsL = -srL*(N[kp,-1]**2 + N0*P0)/(N[kp,-1]+P[kp,-1])**2
        A[1,0]  -= ds0
        A[1,-1] -= dsL
        b[0]  -= sr0*(N[kp, 0]*P[kp, 0]-N0*P0)/(N[kp, 0]+P[kp, 0]) + ds0*P[kp, 0]
        b[-1] -= srL*(N[kp,-1]*P[kp,-1]-N0*P0)/(N[kp,-1]+P[kp,-1]) + dsL*P[kp,-1]
        errP = Solve(A, b, P[kp])

        for n in range(1,L):                    # Solve for E
            A[0,n]  = Lambda*(DP*(P[kp,n]+P[kp,n-1]) + \
                              DN*(N[kp,n]+N[kp,n-1]))/2 + a0
            b[n]    = Lambda*(DP*(P[kp,n]-P[kp,n-1])  - \
                              DN*(N[kp,n]-N[kp,n-1])) - a1*E[k,n] - a2*E[ko,n]
            E[kp,n] = b[n]/A[0,n]

        if ((errN < TOL and errP < TOL)):  break
    return iters+1

@njit(cache=True)
def tEvol(N, P, E, plN, plP, plE, plI, A, b, matPar, simPar):

    Length, Time, L, T, plT, pT, tol, MAX = simPar
    N0   = matPar[:,0]
    P0   = matPar[:,1]
    rate = matPar[:,4]
    itrs = [0]*len(matPar)

    for th in range(len(matPar)):               # Loop over threads
        print("thread: ", th)
        for t in range(T+1):                    # Outer time loop
            if t == 0:                          # Select integration order
                a0 = 1.0; a1 = -1.0; a2 = 0.0   # Euler step
            else:
                a0 = 1.5; a1 = -2.0; a2 = 0.5   # 2nd order implicit
            kp  = (t+1)%3                       # new time
            k   = (t)  %3                       # current time
            ko  = (t-1)%3                       # old time
            par = (a0, a1, a2, k, kp, ko, L, tol, MAX)
            N[th,kp,:] = N[th,k,:]              # Save current values
            P[th,kp,:] = P[th,k,:]
            E[th,kp,:] = E[th,k,:]
            iters = iterate(N[th], P[th], E[th], A[th], b[th], matPar[th], par)

            if itrs[th] < iters:
                itrs[th] = iters
            if iters >= MAX:
                print("NO CONVERGENCE: ", iters, " iterations")
                break
            if t%plT == 0:
                plI[th,t//plT] = rate[th]*(N[th,k]*P[th,k] - N0[th]*P0[th]).sum()
            if t in pT:
                ind = pT.index(t)
                plN[th, ind] = N[th,k]
                plP[th, ind] = P[th,k]
                plE[th, ind] = E[th,k]
    return itrs

def pvSim(matPar, simPar, iniPar):

    # Unpack local parameters
    Length, Time, L, T, plT, pT, tol, MAX = simPar
    dx = Length/L
    dt = Time/T

    # Non dimensionalize variables
    dx3 = dx**3; dtdx = dt/dx; dtdx2 = dtdx/dx
    matPar  = np.array(matPar)
    scales  = np.array([dx3,dx3,dtdx2,dtdx2,dtdx2/dx,dtdx,dtdx,1/dt,1/dt,1/dx])
    matPar *= scales

    # Allocate arrays for each thread
    Threads = len(matPar)
    N   = np.zeros((Threads,3,L))              # Include prior steps and iter
    P   = np.zeros((Threads,3,L))
    E   = np.zeros((Threads,3,L+1))
    A   = np.zeros((Threads,3,L))              # Matrix arrays (inc E)
    b   = np.zeros((Threads,L))                # RHS vectors
    plI = np.zeros((Threads,T//plT+1))         # Arrays for plotting
    plN = np.zeros((Threads,len(pT),L))
    plP = np.zeros((Threads,len(pT),L))
    plE = np.zeros((Threads,len(pT),L+1))

    # Initialization - nodes at 1/2, 3/2 ... L-1/2
    a,l = iniPar
    a  *= dx3
    l  /= dx
    x   = np.arange(L) + 0.5
    dN  = a*np.exp(-x/l)
    N0, P0 = matPar[:,0:2].T
    N[:,0] = np.add.outer(N0, dN)
    P[:,0] = np.add.outer(P0, dN)

    clock0 = time.time()
    itrs = tEvol(N, P, E, plN, plP, plE, plI, A, b, matPar, simPar)
    print("Took {} sec".format(time.time() - clock0))

    plI /= dx**2*dt
    plN /= dx**3
    plP /= dx**3
    plE /= dx

    return itrs, (plN, plP, plE, plI)

if __name__ == "__main__":

    import pickle
    import csv
    simPar, iniPar, matPar = pickle.load(open('i600.pik', 'rb'))

    itrs, pvOut = pvSim(matPar, simPar, iniPar)

    itrs = np.array(itrs).reshape((1, len(itrs)))
    itrs = np.concatenate((itrs.T, matPar), axis=1)
    # with open('itrs.csv', 'w+', newline='') as ofstream:
    #     writer = csv.writer(ofstream, delimiter=',')
    #     writer.writerows(itrs)

    pickle.dump(pvOut, open('o601.pik', 'wb'))

