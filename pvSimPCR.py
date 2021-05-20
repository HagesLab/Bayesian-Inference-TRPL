#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:05:30 2020

@author: tladd
"""
import numpy as np
from numba import cuda, float32 as floatX
import time

@cuda.jit(device=True)
def norm2(A0,A1,A2,b,c,buffer, TPB):
    N = len(b)

    buffer[0] = abs(A1[0]*c[0]+A0[0]*c[1] - b[0])
    buffer[N] = abs(b[0])
    buffer[N-1] = abs(A2[-1]*c[-2]+A1[-1]*c[-1] - b[-1])
    buffer[2*N-1] = abs(b[-1])

    thr = cuda.threadIdx.x
    rf = N // 2
    for i in range(1+thr,N-1,TPB):
        buffer[i] = abs(A2[i]*c[i-1]+A1[i]*c[i]+A0[i]*c[i+1] - b[i])
        buffer[i+N] = abs(b[i])

    cuda.syncthreads()
    while rf >= 1:
        for i in range(thr,rf, TPB):
            buffer[i] = buffer[i] + buffer[i+rf]
            buffer[i+N] = buffer[i+N] + buffer[i+N+rf]
        cuda.syncthreads()
        rf //= 2

    return buffer[0]/buffer[N]

@cuda.jit(device=True)
def pcreduce(ld, d, ud, B, c, buffer, TPB):

    rf = 1
    N = len(ld)

    thr = cuda.threadIdx.x
    while N / rf > 2:
        for i in range(thr, N, TPB):
            buffer[i] = ld[i]
            buffer[i+N] = d[i]
            buffer[i+2*N] = ud[i]
            buffer[i+3*N] = B[i]

        cuda.syncthreads() # Prevent race condition
        for i in range(thr, N, TPB):
            if i >= rf:
                k1 = buffer[i] / buffer[i+N-rf]
                d[i] -= buffer[i+2*N-rf]*k1
                ld[i] = -buffer[i-rf] * k1
                B[i] -= buffer[i+3*N-rf]*k1

            if i < (N - rf):
                k2 = buffer[i+2*N] / buffer[i+N+rf]
                d[i] -= buffer[i+rf]*k2
                ud[i] = -buffer[i+2*N+rf] * k2
                B[i] -= buffer[i+3*N+rf]*k2
        
        cuda.syncthreads()
        rf *= 2
    
    # Solve    
    for i in range(thr, rf, TPB):
        k = ud[i] / d[i+rf]
        c[i] = (B[i] - B[i+rf]*k) / (d[i] - ld[i+rf]*k)
        c[i+rf] = (B[i+rf] - ld[i+rf]*c[i]) / d[i+rf]
        
    return

@cuda.jit(device=True)
def iterate(N, P, E, matPar, par, p, t):

# Unpack local variables
    N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda = matPar
    a0, a1, a2, k, kp, ko, L, tol, MAX, TPB = par

    TOL  = 10.0**(-tol)
    Nk = cuda.shared.array(shape=(SIZ), dtype=floatX)
    Pk = cuda.shared.array(shape=(SIZ), dtype=floatX)
    Ek = cuda.shared.array(shape=(SIZ), dtype=floatX)
    bN = cuda.shared.array(shape=(SIZ), dtype=floatX)
    bP = cuda.shared.array(shape=(SIZ), dtype=floatX)
    bE = cuda.shared.array(shape=(SIZ), dtype=floatX)
    bb = cuda.shared.array(shape=(SIZ), dtype=floatX)
    A0 = cuda.shared.array(shape=(SIZ), dtype=floatX)
    A1 = cuda.shared.array(shape=(SIZ), dtype=floatX)
    A2 = cuda.shared.array(shape=(SIZ), dtype=floatX)
    buffer = cuda.shared.array(shape=(BuSIZ), dtype=floatX)
    th = cuda.threadIdx.x                      # Set thread ID

    for n in range(th, L, TPB):
        Nk[n] = N[k,n]                        # Initialize tmp values
        Pk[n] = P[k,n]
        Ek[n] = E[k,n]
        bN[n] = a1*Nk[n] + a2*N[ko,n]
        bP[n] = a1*Pk[n] + a2*P[ko,n]
        bE[n] = a1*Ek[n] + a2*E[ko,n]

    A0[-1] = 0
    A2[0] = 0

    cuda.syncthreads()
# Iterate outer loop to convergence
    for iters in range(MAX):                    # Up to MAX iterations
        #if p == 1 and t == 0 and iters == 0:
        #    if th == 0: 
        #        print("A0[L-1] before")
        #        print(A0[L-1])
        #        print("A2 before")
        #    print(A2[th])
        #cuda.syncthreads()

        for n in range(1+th, L, TPB):           # Solve for N
            A0[n-1]   = DN*(-Ek[n]/2 - 1)
            A2[n] = DN*(+Ek[n]/2 - 1)
        cuda.syncthreads()
        #if p == 1 and t == 0 and iters == 0:
        #    if th == 0: 
        #        print("A0[L-1] after")
        #        print(A0[-1])
        #        print("A2 after")
        #    print(A2[th])
        #cuda.syncthreads()
        for n in range(th, L, TPB):
            np = Nk[n]*Pk[n] - N0*P0
            tp = Nk[n]*tauP + Pk[n]*tauN
            ds = -rate*Pk[n] - (Pk[n]*tp - tauP*np)/tp**2
            A1[n] = a0 - A0[n-1] - A2[(n+1) % L] - ds
            bb[n] = -(rate + 1/tp)*np - ds*Nk[n] - bN[n]
        cuda.syncthreads()

        ds0 = -sr0*(Pk[ 0]**2 + N0*P0)/(Nk[ 0]+Pk[ 0])**2
        dsL = -srL*(Pk[-1]**2 + N0*P0)/(Nk[-1]+Pk[-1])**2
        A1[0]  -= ds0
        A1[-1] -= dsL
        bb[0]  -= sr0*(Nk[ 0]*Pk[ 0]-N0*P0)/(Nk[ 0]+Pk[ 0]) + ds0*Nk[ 0]
        bb[-1] -= srL*(Nk[-1]*Pk[-1]-N0*P0)/(Nk[-1]+Pk[-1]) + dsL*Nk[-1]

        errN = norm2(A0,A1,A2,bb,Nk,buffer, TPB)
        cuda.syncthreads()

        pcreduce(A2, A1, A0, bb, Nk, buffer, TPB)

        cuda.syncthreads()

        for n in range(1+th, L, TPB):                    # Solve for P
            A0[n-1]   = DP*(+Ek[n]/2 - 1)
            A2[n] = DP*(-Ek[n]/2 - 1)
        cuda.syncthreads()
        for n in range(th, L, TPB):
            np = Nk[n]*Pk[n] - N0*P0
            tp = Nk[n]*tauP + Pk[n]*tauN
            ds = -rate*Nk[n] - (Nk[n]*tp - tauN*np)/tp**2
            A1[n] = a0 - A0[n-1] - A2[(n+1) % L] - ds
            bb[n] = -(rate + 1/tp)*np - ds*Pk[n] - bP[n]
        cuda.syncthreads()
        ds0 = -sr0*(Nk[ 0]**2 + N0*P0)/(Nk[ 0]+Pk[ 0])**2
        dsL = -srL*(Nk[-1]**2 + N0*P0)/(Nk[-1]+Pk[-1])**2
        A1[0]  -= ds0
        A1[-1] -= dsL
        bb[0]  -= sr0*(Nk[ 0]*Pk[ 0]-N0*P0)/(Nk[ 0]+Pk[ 0]) + ds0*Pk[ 0]
        bb[-1] -= srL*(Nk[-1]*Pk[-1]-N0*P0)/(Nk[-1]+Pk[-1]) + dsL*Pk[-1]

        errP = norm2(A0,A1,A2,bb,Pk, buffer, TPB)
        cuda.syncthreads()
        pcreduce(A2, A1, A0, bb, Pk, buffer, TPB)
        cuda.syncthreads()

        for n in range(1+th, L, TPB):                    # Solve for E
            A1[n] = Lambda*(DP*(Pk[n]+Pk[n-1]) + DN*(Nk[n]+Nk[n-1]))/2 + a0
            bb[n] = Lambda*(DP*(Pk[n]-Pk[n-1]) - DN*(Nk[n]-Nk[n-1])) - bE[n]
            Ek[n] = bb[n]/A1[n]          
        cuda.syncthreads()
        if ((errN < TOL and errP < TOL)):  break    
    
    for n in range(th, L, TPB):
        N[kp,n] = Nk[n]                         # Copy back tmp values
        P[kp,n] = Pk[n]
        E[kp,n] = Ek[n]
    cuda.syncthreads()
    #if t == 0 and p == 1:                
    #   if th == 0: 
    #       print("Iters=")
    #       print(iters)
    #cuda.syncthreads()


    return iters+1

@cuda.jit(device=False)
def tEvol(N, P, E, plN, plP, plE, plI, matPar, simPar, gridPar, race):
    L, T, tol, MAX, plT = simPar[:5]
    pT   = simPar[5:]
    N0   = matPar[:,0]
    P0   = matPar[:,1]
    rate = matPar[:,4]
    BPG, TPB = gridPar
    ind  = 0
    for t in range(T+1):                            # Outer time loop
    #for t in range(10):
        #if t%100 ==0 and cuda.grid(1) == 0:
        #    print('time: ', t)
        if t == 0:                                  # Select integration order
            a0 = 1.0; a1 = -1.0; a2 = 0.0           # Euler step
        else:
            a0 = 1.5; a1 = -2.0; a2 = 0.5           # 2nd order implicit
        kp  = (t+1)%3                               # new time
        k   = (t)  %3                               # current time
        ko  = (t-1)%3                               # old time
        par = (a0, a1, a2, k, kp, ko, L, tol, MAX, TPB)
        for p in range(cuda.blockIdx.x, len(matPar), BPG): # Use blocks to look over param sets
        #for blk in range(1):
            #if t==0 and cuda.grid(1) == 0:
            #    print('block', blk, 'Starting p: ', blk*BPG)
            if cuda.threadIdx.x == 0: race[p] += 1
            #if p == 0 and cuda.threadIdx.x == 0: 
            #    print("Running #0")
            #cuda.syncthreads()
            
            iters = iterate(N[p], P[p], E[p], matPar[p], par, p, t)
            #if t == 0 and p == 1 and cuda.threadIdx.x == 0: 
            #    print("Ran #1")
            #    print("k:")
            #    print(k)
            #    print("iters")
            #    print(iters)
            #    print("N[p,kp,0]:")
            #    print(N[p,kp,0])
            cuda.syncthreads()
            #else: iters = 0
            if iters >= MAX:
                if cuda.threadIdx.x == 0:
                    print('NO CONVERGENCE: ', \
                      'Block ', p, 'simtime ', t, iters, ' iterations')
                break
            
            if t%plT == 0:
                Sum = 0
                #if t == 0 and p == 1 and cuda.threadIdx.x == 0:
                #    print("k again:")
                #    print(k)
                #    print("N[p,kp,0]")
                #    print(N[p,kp,0])
                #cuda.syncthreads()
                for n in range(L):
                    Sum += N[p,k,n]*P[p,k,n]-N0[p]*P0[p]
                plI[p,t//plT] = rate[p]*Sum
            

            # Record specified timesteps, for debug mode
            #if t == pT[ind]:
            #    for n in range(L):
            #        plN[p,ind,n] = N[p,k,n] 
            #        plP[p,ind,n] = P[p,k,n]
            #        plE[p,ind,n] = E[p,k,n]
        #if t == pT[ind]:  ind += 1
        
        cuda.syncthreads()
        """
        if t%plT == 0:
            for p in range(cuda.grid(1), len(matPar), cuda.gridsize(1)):
                Sum = 0
                for n in range(L):
                    Sum += N[p,k,n]*P[p,k,n]-N0[p]*P0[p]
                plI[p,t//plT] = rate[p]*Sum
            cuda.syncthreads()
        """
    # Record last two timesteps
    th = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    for p in range(th,len(N), cuda.gridsize(1)):
        for n in range(len(N[0,0])):
            plN[p,1,n] = N[p,k,n]
            plN[p,0,n] = N[p,ko,n] 
            plP[p,1,n] = P[p,k,n]
            plP[p,0,n] = P[p,ko,n]

        for n in range(len(E[0,0])):
            plE[p,1,n] = E[p,k,n]
            plE[p,0,n] = E[p,ko,n]
          

def pvSim(matPar, simPar, iniPar, TPB, BPG, max_sims_per_block=1, init_mode="exp"):
    print("Solver called")
    print((TPB, BPG))
    # Unpack local parameters
    Length, Time, L, T, plT, pT, tol, MAX = simPar
    dx = Length/L
    dt = Time/T
    simPar = (L, T, tol, MAX, plT, *pT)
    gridPar = (BPG, *TPB)
    global SIZ 
    SIZ = L
    global BuSIZ
    BuSIZ = int(SIZ)*4
    global MSPB
    MSPB = max_sims_per_block

    # Non dimensionalize variables
    dx3 = dx**3; dtdx = dt/dx; dtdx2 = dtdx/dx
    matPar  = np.array(matPar)
    scales  = np.array([dx3,dx3,dtdx2,dtdx2,dtdx2/dx,dtdx,dtdx,1/dt,1/dt,1/dx])
    matPar *= scales

    # Allocate arrays for each thread
    Threads = len(matPar)                      # Number of threads on GPU


    plI_main = np.zeros((Threads,(T//plT+1) * len(iniPar)))         # Arrays for plotting
    count = 0
    for ic in iniPar:

        N   = np.zeros((Threads,3,L))              # Include prior steps and iter
        P   = np.zeros((Threads,3,L))
        E   = np.zeros((Threads,3,L+1))
        #plN = np.zeros((Threads,len(pT),L))
        #plP = np.zeros((Threads,len(pT),L))
        #plE = np.zeros((Threads,len(pT),L+1))
        plI = np.zeros((Threads,T//plT+1))
        plN = np.zeros((Threads, 2, L))
        plP = np.zeros((Threads, 2, L))
        plE = np.zeros((Threads, 2, L+1))

        if init_mode == "exp":
            # Initialization - nodes at 1/2, 3/2 ... L-1/2
            a,l = ic
            a  *= dx3
            l  /= dx
            x   = np.arange(L) + 0.5
            dN  = a *np.exp(-x/l)
    
        elif init_mode == "points":
            dN = ic * dx3

        N0, P0 = matPar[:,0:2].T
        N[:,0] = np.add.outer(N0, dN)
        P[:,0] = np.add.outer(P0, dN)
        #print("Incoming N[1]:", N[1])
        clock0 = time.time()
        devN = cuda.to_device(N)
        devP = cuda.to_device(P)
        devE = cuda.to_device(E)
        devpN = cuda.to_device(plN)
        devpP = cuda.to_device(plP)
        devpE = cuda.to_device(plE)
        devpI = cuda.to_device(plI)
        devm = cuda.to_device(matPar)
        devs = cuda.to_device(simPar)
        devg = cuda.to_device(gridPar)
        print("Loading data took {} sec".format(time.time() - clock0))
        race = np.zeros(len(matPar))
        drace = cuda.to_device(race)
        clock0 = time.time()
        tEvol[BPG,TPB](devN, devP, devE, devpN, devpP, devpE, devpI, devm, devs, devg, drace)
        cuda.synchronize()
        print("tEvol took {} sec".format(time.time() - clock0))
    
        clock0 = time.time()
        plI = devpI.copy_to_host()
        plN = devpN.copy_to_host()
        plP = devpP.copy_to_host()
        plE = devpE.copy_to_host()
        print("Copy back took {} sec".format(time.time() - clock0))
        race = drace.copy_to_host()
        print(race[race != T//plT+1])
        plI_main[:, (T//plT+1)*count:(T//plT+1)*(count+1)] = plI
        count += 1
    # Re-dimensionalize
    plI_main /= dx**2*dt
    plN /= dx**3
    plP /= dx**3
    plE /= dx
    print(plI_main[:,0:6])
    print(list(np.sum(plI_main, axis=1)))
    return (plN, plP, plE, plI_main)

if __name__ == "__main__":

    import pickle
    cuda.detect()
    device = cuda.get_current_device()
    SM_count = getattr(device, "MULTIPROCESSOR_COUNT")
    print("SMs: ", SM_count)
    dir = r"/blue/c.hages/cfai2304/"
    fname = 'ipvtest.pik'
    oname = 'o602.pik'
    print("Reading input ", dir + fname)
    simPar, iniPar, matPar = pickle.load(open(dir + fname, 'rb'))
    TPB = simPar[2]                               # Define # Threads per block: one per node 
    BPG = SM_count * 2                                    # Define # Blocks: multiple of # available SMs
    pvOut = pvSim(matPar, simPar, iniPar, TPB, BPG)
    #pickle.dump(pvOut, open(dir + oname, 'wb'))
    print("Wrote results to ", dir + oname)
