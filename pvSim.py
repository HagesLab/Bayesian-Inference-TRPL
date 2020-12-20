 # -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:57:05 2020

@author: cfai2
"""

import math, numpy as np
from numba import njit
import time

@njit(cache=True)
def Sum(X):
    Sum = 0
    for x in X:
        Sum += x
    return Sum

@njit(cache=True)
def Norm(x, xO, tol):
    Sum = SumX = 0
    for n in range(len(x)):
        Sum  += (x[n]-xO[n])*(x[n]-xO[n])
        SumX += x[n]*x[n]
    return math.sqrt(Sum/(SumX+tol))

@njit(cache=True)
def Copy(a, b):

    n = len(a)
    for k in range(n):
        b[k] = a[k]

@njit(cache=True)
def Solve(A, b, c):

    n = len(b)
    for k in range(n-1):                            # Elimination
        q = A[2,k]/A[1,k]
        A[1,k+1] -= q*A[0,k+1]
        b[k+1]   -= q*b[k]

    c[-1] = b[-1]/A[1,-1]                           # Backsubstitution
    for k in range(n-2,-1,-1):
        c[k] = (b[k]-A[0,k+1]*c[k+1])/A[1,k]

@njit(cache=True)
def timeSteps(N, P, E, A, b, matPar, N0, P0, t, tol, MAX):

# Unpack local variables
    TOL = 10.0**(-tol)
    N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda = matPar
    if t == 1:                                  # Select integration order
        a0 = 1.0; a1 = -1.0; a2 = 0.0           # Euler step
    else:
        a0 = 1.5; a1 = -2.0; a2 = 0.5           # 2nd order implicit
    k  = (t)%3
    ko = (t-1)%3
    kp = (t+1)%3
    L  = len(N[0])                              # Get length
    N[kp,:] = N[k,:]                            # Save current values
    P[kp,:] = P[k,:]
    E[kp,:] = E[k,:]

    for iters in range(MAX):                    # Up to MAX iterations
        for n in range(1,L):                    # Solve for N
            A[0,n]   = DN*(-E[kp,n]/2 - 1)
            A[2,n-1] = DN*(+E[kp,n]/2 - 1)
        for n in range(L):
            N[-1,n]  = N[kp,n]                  # Save previous iter
            dRdn = -rate*P[kp,n] - ((P[kp,n])*(tauP*N[kp,n] + tauN*P[kp,n]) - tauP*(N[kp,n]*P[kp,n] - N0*P0))/(tauP*N[kp,n] + tauN*P[kp,n]) ** 2
            A[1,n]   = a0 - A[0,n] - A[2,n] - dRdn
            b[n]   = -(rate + 1/(tauP*N[kp,n] + tauN*P[kp,n])) * \
                      (N[kp,n]*P[kp,n] - N0*P0) - dRdn * N[kp,n] - a1*N[k,n] - a2*N[ko,n]

        ds0dn = -sr0*(P[kp,0]**2 + N0*P0) / (N[kp,0]+P[kp,0]) ** 2
        dsLdn = -srL*(P[kp,-1]**2 + N0*P0) / (N[kp,-1]+P[kp,-1]) ** 2
        A[1,0] -= ds0dn
        A[1,-1] -= dsLdn
        b[0]  -= sr0*(N[kp,0]*P[kp,0]  - N0*P0)/(N[kp,0]+P[kp,0]) + ds0dn * N[kp,0]
        b[-1] -= srL*(N[kp,-1]*P[kp,-1] - N0*P0)/(N[kp,-1]+P[kp,-1]) + dsLdn * N[kp,-1]

        Solve(A, b, N[kp])

        for n in range(1,L):                    # Solve for P
            A[0,n]   = DP*(+E[kp,n]/2 - 1)
            A[2,n-1] = DP*(-E[kp,n]/2 - 1)
        for n in range(L):
            P[-1,n]  = P[kp,n]                  # Save previous iter
            dRdp = -rate*N[-1,n] - ((N[-1,n])*(tauP*N[-1,n] + tauN*P[kp,n]) - tauN*(N[-1,n]*P[kp,n] - N0*P0))/(tauP*N[-1,n] + tauN*P[kp,n]) ** 2
            A[1,n]   = a0 - A[0,n] - A[2,n] - dRdp
            b[n]   = -(rate + 1/(tauP*N[-1,n] + tauN*P[kp,n])) * \
                      (N[-1,n]*P[kp,n] - N0*P0) - dRdp * P[kp,n] - a1*P[k,n] - a2*P[ko,n]

        ds0dp = -sr0*(N[-1,0]**2 + N0*P0) / (N[-1,0]+P[kp,0]) ** 2
        dsLdp = -srL*(N[-1,-1]**2 + N0*P0) / (N[-1,-1]+P[kp,-1]) ** 2
        A[1,0] -= ds0dp
        A[1,-1] -= dsLdp

        b[0]  -= sr0*(N[-1,0]*P[kp,0]  - N0*P0)/(N[-1,0]+P[kp,0]) + ds0dp * P[kp,0]
        b[-1] -= srL*(N[-1,-1]*P[kp,-1] - N0*P0)/(N[-1,-1]+P[kp,-1]) + dsLdp * P[kp,-1]
        Solve(A, b, P[kp])

        for n in range(1,L):                    # Solve for E
            E[-1,n] = E[kp,n]
            A[0,n]  = Lambda*(DP*(P[kp,n]+P[kp,n-1]) + \
                              DN*(N[kp,n]+N[kp,n-1]))/2 + a0
            b[n]    = Lambda*(DP*(P[kp,n]-P[kp,n-1]) - \
                              DN*(N[kp,n]-N[kp,n-1])) - \
                     a1*E[k,n] - a2*E[ko,n]
            E[kp,n] = b[n]/A[0,n]

        normN = Norm(N[kp], N[-1], TOL/10)
        normP = Norm(P[kp], P[-1], TOL/10)
        #normE = Norm(E[kp], E[-1], TOL/10)
#        print(iters, normN, normP, normE)
        if ((normN < TOL and normP < TOL)):  break
    return iters+1

@njit(cache=True)
def tEvol(N, P, E, plN, plP, plE, plI, A, b, matPar, \
          Length, Time, L, T, plT, pT, tol, MAX):
    N0   = matPar[:,0]
    P0   = matPar[:,1]
    rate = matPar[:,4]
    fail_states = [0]*len(matPar)
    # Threads = [449, 471, 481, 633, 758, 764, 768, 770, 776, 778, 782, 800, \
    #             804, 808, 920, 926, 928, 932, 934, 940, 944, 946, 952, 955, \
    #             956, 964, 968, 970, 1094, 1099, 1110, 1287, 1290, 1294, 2056, \
    #           2062, 2064, 2066, 2068, 2080, 2084, 2102, 2214, 2226, 2236, \
    #           2244, 2246, 2248, 2252, 2254, 2258, 2259, 2262, 2264, 2266, 2582]
    #Threads = [920, 926, 2066, 2084, 2244]
    #for thr in Threads:
    #for thr in range(323,325):
    for thr in range(len(matPar)):              # Loop over thrs

        print("thread: ", thr)
        for t in range(1,T+1):                  # Outer time loop
            iters = timeSteps(N[thr], P[thr], E[thr], A[thr], b[thr], matPar[thr], \
                             N0[thr], P0[thr], t, tol, MAX)
            if iters >= MAX:
                print("NO CONVERGENCE: ", iters, " iterations")
                fail_states[thr] = -1
                break
            # else:
            #     print("Converged after: ", iter, " iterations")

            if t%plT == 0:
                plI[thr,t//plT] = Sum(rate[thr]*(N[thr,0]*P[thr,0] - \
                                                 N0[thr]*P0[thr]))
            if t in pT:
                ind = pT.index(t)
                Copy(N[thr,t%3], plN[thr,ind])
                Copy(P[thr,t%3], plP[thr,ind])
                Copy(E[thr,t%3], plE[thr,ind])


    return fail_states

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
    N   = np.zeros((Threads,4,L))              # Include prior steps and iter
    P   = np.zeros((Threads,4,L))
    E   = np.zeros((Threads,4,L+1))
    A   = np.zeros((Threads,3,L))              # Matrix arrays (inc E)
    b   = np.zeros((Threads,L))                # RHS vectors
    plI = np.zeros((Threads,T//plT+1))         # Arrays for plotting
    plN = np.zeros((Threads,len(pT),L))
    plP = np.zeros((Threads,len(pT),L))
    plE = np.zeros((Threads,len(pT),L+1))

    # Initialization - nodes at 1/2, 3/2 ... L-1/2
    a, l = iniPar
    a  *= dx3
    l  /= dx
    N0   = matPar[:,0]
    P0   = matPar[:,1]
    rate = matPar[:,4]
    x  = np.arange(L) + 0.5
    dN = a * np.exp(-x/l)
    N0, P0 = matPar[:,0:2].T
    N[:,0] = np.add.outer(N0, dN)
    P[:,0] = np.add.outer(P0, dN)
    N[:,1] = N[:,0].copy()
    P[:,1] = P[:,0].copy()
    plN[:,0] = N[:,0]
    plP[:,0] = P[:,0]
    plE[:,0] = E[:,0]
    plI[:,0] = rate[:]*((N[:,0]*P[:,0]).T - N0*P0).sum(axis=0)

    clock0 = time.time()
    fail_states = tEvol(N, P, E, plN, plP, plE, plI, A, b, matPar, *simPar)
    print("Took {} sec".format(time.time() - clock0))

    plI *= dx**4/dt
    plN /= dx**3
    plP /= dx**3
    plE /= dx

    return fail_states, (plN, plP, plE, plI)

if __name__ == "__main__":

    import pickle
    import csv
    simPar, iniPar, matPar = pickle.load(open('hagesInputs300.pik', 'rb'))
    #matPar = np.array(matPar[3457]).reshape((1, len(matPar[3457])))
    pTh = -1
    fail_states, pvOut = pvSim(matPar, simPar, iniPar)

    fail_states = np.array(fail_states).reshape((1, len(fail_states)))
    # fail_states = np.concatenate((fail_states.T, matPar), axis=1)
    # with open('fail3.csv', 'w+', newline='') as ofstream:
    #     writer = csv.writer(ofstream, delimiter=',')
    #     writer.writerows(fail_states)

    pickle.dump(pvOut, open('hagesOut300.pik', 'wb'))

