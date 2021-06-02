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
def norm2(A0,A1,A2,b,c,buffer, err, TPB):
    N = len(b)
    num_sims = len(A0[0])
    thr = cuda.threadIdx.x
    for y in range(thr, num_sims, TPB):
        buffer[0, y] = abs(A1[0, y]*c[0, y]+A0[0, y]*c[1, y] - b[0, y])
        buffer[N, y] = abs(b[0, y])
        buffer[N-1, y] = abs(A2[-1, y]*c[-2, y]+A1[-1, y]*c[-1, y] - b[-1, y])
        buffer[2*N-1, y] = abs(b[-1, y])
    cuda.syncthreads()
    rf = N // 2
    for i in range(1+thr,N-1,TPB):
        for y in range(num_sims):
            buffer[i, y] = abs(A2[i, y]*c[i-1, y]+A1[i, y]*c[i, y]+A0[i, y]*c[i+1, y] - b[i, y])
            buffer[i+N, y] = abs(b[i, y])

    cuda.syncthreads()
    while rf >= 1:
        for i in range(thr,rf, TPB):
            for y in range(num_sims):
                buffer[i, y] = buffer[i, y] + buffer[i+rf, y]
                buffer[i+N, y] = buffer[i+N, y] + buffer[i+N+rf, y]
        cuda.syncthreads()
        rf //= 2
    for y in range(thr, num_sims, TPB):
        err[y] = buffer[0, y]/buffer[N, y]

@cuda.jit(device=True)
def pcreduce(ld, d, ud, B, c, buffer, TPB):

    rf = 1
    N = len(ld)
    num_sims = len(ld[0])
    thr = cuda.threadIdx.x
    while N / rf > 2:
        for i in range(thr, N, TPB):
            for y in range(num_sims):
                buffer[i, y] = ld[i, y]
                buffer[i+N, y] = d[i, y]
                buffer[i+2*N, y] = ud[i, y]
                buffer[i+3*N, y] = B[i, y]

        cuda.syncthreads() # Prevent race condition
        for i in range(thr, N, TPB):
            for y in range(num_sims):
                if i >= rf:
                    k1 = buffer[i, y] / buffer[i+N-rf, y]
                    d[i, y] -= buffer[i+2*N-rf, y]*k1
                    ld[i, y] = -buffer[i-rf, y] * k1
                    B[i, y] -= buffer[i+3*N-rf, y]*k1

                if i < (N - rf):
                    k2 = buffer[i+2*N, y] / buffer[i+N+rf, y]
                    d[i, y] -= buffer[i+rf, y]*k2
                    ud[i, y] = -buffer[i+2*N+rf, y] * k2
                    B[i, y] -= buffer[i+3*N+rf, y]*k2
        
        cuda.syncthreads()
        rf *= 2
    
    # Solve    
    for i in range(thr, rf, TPB):
        for y in range(num_sims):
            k = ud[i, y] / d[i+rf, y]
            c[i, y] = (B[i, y] - B[i+rf, y]*k) / (d[i, y] - ld[i+rf,y]*k)
            c[i+rf, y] = (B[i+rf, y] - ld[i+rf, y]*c[i, y]) / d[i+rf, y]
        
    return

@cuda.jit(device=True)
def shared_array_max(arr):
    m = arr[0]
    for i in range(1,len(arr)):
        if arr[i] > m:
            m = arr[i]

    return m


@cuda.jit(device=True)
def iterate(N, P, E, matPar, par, p, t):

# Unpack local variables
    N0 = matPar[:,0]
    P0 = matPar[:,1]
    DN = matPar[:,2]
    DP = matPar[:,3]
    rate = matPar[:,4]
    sr0 = matPar[:,5]
    srL = matPar[:,6]
    tauN = matPar[:,7]
    tauP = matPar[:,8]
    Lambda = matPar[:,9]
    #N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, Lambda = matPar
    a0, a1, a2, k, kp, ko, L, tol, MAX, TPB = par
    num_sims = len(N)
    TOL  = 10.0**(-tol)
    Nk = cuda.shared.array(shape=(SIZ, MSPB), dtype=floatX)
    Pk = cuda.shared.array(shape=(SIZ, MSPB), dtype=floatX)
    Ek = cuda.shared.array(shape=(SIZ, MSPB), dtype=floatX)
    bN = cuda.shared.array(shape=(SIZ, MSPB), dtype=floatX)
    bP = cuda.shared.array(shape=(SIZ, MSPB), dtype=floatX)
    bE = cuda.shared.array(shape=(SIZ, MSPB), dtype=floatX)
    bb = cuda.shared.array(shape=(SIZ, MSPB), dtype=floatX)
    A0 = cuda.shared.array(shape=(SIZ, MSPB), dtype=floatX)
    A1 = cuda.shared.array(shape=(SIZ, MSPB), dtype=floatX)
    A2 = cuda.shared.array(shape=(SIZ, MSPB), dtype=floatX)
    buffer = cuda.shared.array(shape=(BuSIZ, MSPB), dtype=floatX)
    errN = cuda.shared.array(shape=(MSPB), dtype=floatX)
    errP = cuda.shared.array(shape=(MSPB), dtype=floatX)
    th = cuda.threadIdx.x                      # Set thread ID

    for n in range(th, L, TPB):
        for y in range(num_sims):
            Nk[n, y] = N[y,k,n]                        # Initialize tmp values
            Pk[n, y] = P[y,k,n]
            Ek[n, y] = E[y,k,n]
            bN[n, y] = a1*Nk[n, y] + a2*N[y,ko,n]
            bP[n, y] = a1*Pk[n, y] + a2*P[y,ko,n]
            bE[n, y] = a1*Ek[n, y] + a2*E[y,ko,n]

    for y in range(th, num_sims, TPB):
        A0[-1, y] = 0
        A2[0, y] = 0
        errN[y] = 0
        errP[y] = 0

    cuda.syncthreads()
# Iterate outer loop to convergence
    for iters in range(MAX):                    # Up to MAX iterations
        for n in range(1+th, L, TPB):           # Solve for N
            for y in range(num_sims):
                A0[n-1, y]   = DN[y]*(-Ek[n, y]/2 - 1)
                A2[n, y] = DN[y]*(+Ek[n, y]/2 - 1)
        cuda.syncthreads()
        #if p == 1 and t == 0 and iters == 0:
        #    if th == 0: 
        #        print("A0[L-1] after")
        #        print(A0[-1])
        #        print("A2 after")
        #    print(A2[th])
        #cuda.syncthreads()
        for n in range(th, L, TPB):
            for y in range(num_sims):
                tp = Nk[n, y]*tauP[y] + Pk[n, y] * tauN[y]
                np = Nk[n, y]*Pk[n, y] - N0[y]*P0[y]
                ds = -rate[y]*Pk[n, y] - (Pk[n, y]*tp - tauP[y]*np)/tp**2
                A1[n, y] = a0 - A0[n-1, y] - A2[(n+1) % L, y] - ds
                #if iters == 0 and p == 0 and t == 0:
                #    if th == 0:
                #        print("y")
                #        print(y)
                #        print("A1")
                #    print(A1[th,y])
                bb[n, y] = -(rate[y] + 1/tp)*np - ds*Nk[n, y] - bN[n, y]
        cuda.syncthreads()

        for y in range(th, num_sims, TPB):
            ds0 = -sr0[y]*(Pk[ 0, y]**2 + N0[y]*P0[y])/(Nk[ 0, y]+Pk[ 0, y])**2
            dsL = -srL[y]*(Pk[-1, y]**2 + N0[y]*P0[y])/(Nk[-1, y]+Pk[-1, y])**2
            A1[0, y]  -= ds0
            A1[-1, y] -= dsL
            bb[0, y]  -= sr0[y]*(Nk[ 0, y]*Pk[ 0, y]-N0[y]*P0[y])/(Nk[ 0, y]+Pk[ 0, y]) + ds0*Nk[ 0, y]
            bb[-1, y] -= srL[y]*(Nk[-1, y]*Pk[-1, y]-N0[y]*P0[y])/(Nk[-1, y]+Pk[-1, y]) + dsL*Nk[-1, y]

        cuda.syncthreads()
        norm2(A0,A1,A2,bb,Nk,buffer, errN,TPB)
        cuda.syncthreads()

        pcreduce(A2, A1, A0, bb, Nk, buffer, TPB)

        cuda.syncthreads()

        for n in range(1+th, L, TPB):                    # Solve for P
            for y in range(num_sims):
                A0[n-1, y]   = DP[y]*(+Ek[n, y]/2 - 1)
                A2[n, y] = DP[y]*(-Ek[n, y]/2 - 1)
        cuda.syncthreads()
        for n in range(th, L, TPB):
            for y in range(num_sims):
                np = Nk[n, y]*Pk[n, y] - N0[y]*P0[y]
                tp = Nk[n, y]*tauP[y] + Pk[n, y]*tauN[y]
                ds = -rate[y]*Nk[n, y] - (Nk[n, y]*tp - tauN[y]*np)/tp**2
                A1[n, y] = a0 - A0[n-1, y] - A2[(n+1) % L, y] - ds
                bb[n, y] = -(rate[y] + 1/tp)*np - ds*Pk[n, y] - bP[n, y]
        cuda.syncthreads()
        for y in range(th, num_sims, TPB):
            ds0 = -sr0[y]*(Nk[ 0, y]**2 + N0[y]*P0[y])/(Nk[ 0, y]+Pk[ 0, y])**2
            dsL = -srL[y]*(Nk[-1, y]**2 + N0[y]*P0[y])/(Nk[-1, y]+Pk[-1, y])**2
            A1[0, y]  -= ds0
            A1[-1, y] -= dsL
            bb[0, y]  -= sr0[y]*(Nk[ 0, y]*Pk[ 0, y]-N0[y]*P0[y])/(Nk[ 0, y]+Pk[ 0, y]) + ds0*Pk[ 0, y]
            bb[-1, y] -= srL[y]*(Nk[-1, y]*Pk[-1, y]-N0[y]*P0[y])/(Nk[-1, y]+Pk[-1, y]) + dsL*Pk[-1, y]
        cuda.syncthreads()
        norm2(A0,A1,A2,bb,Pk, buffer, errP,TPB)
        cuda.syncthreads()
        pcreduce(A2, A1, A0, bb, Pk, buffer, TPB)
        cuda.syncthreads()

        for n in range(1+th, L, TPB):                    # Solve for E
            for y in range(num_sims):
                A1[n, y] = Lambda[y]*(DP[y]*(Pk[n, y]+Pk[n-1, y]) + DN[y]*(Nk[n, y]+Nk[n-1, y]))/2 + a0
                bb[n, y] = Lambda[y]*(DP[y]*(Pk[n, y]-Pk[n-1, y]) - DN[y]*(Nk[n, y]-Nk[n-1, y])) - bE[n, y]
                Ek[n, y] = bb[n, y]/A1[n, y]          
        cuda.syncthreads()

        #if (errN[0] < TOL and errP[0] < TOL): break
        max_errN = shared_array_max(errN)
        max_errP = shared_array_max(errP)
        cuda.syncthreads()
        if (max_errN < TOL and max_errP < TOL): break
    
    for n in range(th, L, TPB):
        for y in range(num_sims):
            N[y,kp,n] = Nk[n, y]                         # Copy back tmp values
            P[y,kp,n] = Pk[n, y]
            E[y,kp,n] = Ek[n, y]
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
        for p in range(cuda.blockIdx.x*MSPB, len(matPar), BPG*MSPB): # Use blocks to look over param sets
        #for blk in range(1):
            #if t==0 and cuda.grid(1) == 0:
            #    print('block', blk, 'Starting p: ', blk*BPG)
            if cuda.threadIdx.x == 0: race[p] += 1
            #if p == 0 and cuda.threadIdx.x == 0: 
            #    print("Running #0")
            #cuda.syncthreads()

            iters = iterate(N[p:p+MSPB], P[p:p+MSPB], E[p:p+MSPB], matPar[p:p+MSPB], par, p, t)

            cuda.syncthreads()
            if iters >= MAX:
                if cuda.threadIdx.x == 0:
                    print('NO CONVERGENCE: ', \
                      'Block ', p, 'simtime ', t, iters, ' iterations')
                race[-1] = 1
                break
                
            
            if t%plT == 0:
                for y in range(p, p+len(N[p:p+MSPB])):
                    Sum = 0
                    for n in range(L):
                        Sum += N[y,k,n]*P[y,k,n]-N0[y]*P0[y]
                    plI[y,t//plT] = rate[y]*Sum

            # Record specified timesteps, for debug mode
            #if t == pT[ind]:
            #    for n in range(L):
            #        plN[p,ind,n] = N[p,k,n] 
            #        plP[p,ind,n] = P[p,k,n]
            #        plE[p,ind,n] = E[p,k,n]
        #if t == pT[ind]:  ind += 1
        if race[-1]:
            if cuda.threadIdx.x == 0: print("Block ", cuda.blockIdx.x, " stopping due to nonconvergence:")
            break
        cuda.syncthreads()

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
          

def pvSim(plI_main, plN_main, plP_main, plE_main, matPar, simPar, iniPar, TPB, BPG, max_sims_per_block=1, init_mode="exp"):
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


    #plI_main = np.zeros((Threads,(T//plT+1) * len(iniPar)))         # Arrays for plotting

    N   = np.zeros((Threads,3,L))              # Include prior steps and iter
    P   = np.zeros((Threads,3,L))
    E   = np.zeros((Threads,3,L+1))
    #plN = np.zeros((Threads,len(pT),L))
    #plP = np.zeros((Threads,len(pT),L))
    #plE = np.zeros((Threads,len(pT),L+1))
    #plI = np.zeros((Threads,T//plT+1))

    if init_mode == "exp":
        # Initialization - nodes at 1/2, 3/2 ... L-1/2
        a,l = iniPar
        a  *= dx3
        l  /= dx
        x   = np.arange(L) + 0.5
        dN  = a *np.exp(-x/l)
    
    elif init_mode == "points":
        dN = iniPar * dx3
    elif init_mode == "continue":
        pass

    N0, P0 = matPar[:,0:2].T
    N[:,0] = np.add.outer(N0, dN)
    P[:,0] = np.add.outer(P0, dN)
    #print("Incoming N[0]:", N[0])
    clock0 = time.time()
    devN = cuda.to_device(N)
    devP = cuda.to_device(P)
    devE = cuda.to_device(E)
    devpN = cuda.to_device(plN_main)
    devpP = cuda.to_device(plP_main)
    devpE = cuda.to_device(plE_main)
    devpI = cuda.to_device(plI_main)
    devm = cuda.to_device(matPar)
    devs = cuda.to_device(simPar)
    devg = cuda.to_device(gridPar)
    print("Loading data took {} sec".format(time.time() - clock0))
    race = np.zeros(len(matPar) + 1)
    drace = cuda.to_device(race)
    clock0 = time.time()
    tEvol[BPG,TPB](devN, devP, devE, devpN, devpP, devpE, devpI, devm, devs, devg, drace)
    cuda.synchronize()
    print("tEvol took {} sec".format(time.time() - clock0))
    clock0 = time.time()
    plI_main[:] = devpI.copy_to_host()
    plN_main[:] = devpN.copy_to_host()
    plP_main[:] = devpP.copy_to_host()
    plE_main[:] = devpE.copy_to_host()
    print("Copy back took {} sec".format(time.time() - clock0))
    race = drace.copy_to_host()
    print(race[race != T//plT+1])

    # Re-dimensionalize
    plI_main /= dx**2*dt
    plN_main /= dx**3
    plP_main /= dx**3
    plE_main /= dx
    print(plI_main[:,0:6])
    print(list(np.sum(plI_main, axis=1)))
    return

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
