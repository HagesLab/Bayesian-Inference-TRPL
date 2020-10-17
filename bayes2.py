#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:33:18 2020

@author: tladd
"""
## Define constants
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

import numpy as np

def indexGrid(N, refs):                        # Arrays of cell coordinates
    cN  = N.copy()                             # Copy of cell indexes
    K   = len(refs)                            # Num params
    M   = len(refs[0])
    pN  = np.ones(M, int)
    indexes = np.zeros((len(N),M), int)        # Initialize indexes
    for k in range(K):                         # Loop over refinement levels
        ref = refs[K-k-1]                      # Loop backwards
        ind = []                               # Indexes for current level
        for m in range(len(ref)):              # Loop over directions
            ind.append(cN%ref[m])              # Index for this direction
            cN //= ref[m]                      # Divide out fom N
        indexes += np.array(ind).T*pN          # Accumulate indexes from level
        pN  *= refs[k]                         # Update mutipliers
    return indexes

def indexN(indexes, refs):                     # Arrays of cell indexes
    ind = indexes.copy()                       # Copy of coordinates
    K   = len(refs)                            # Num params
    pN  = np.prod(refs, axis=0)                # Scale for indexes
    N   = np.zeros(len(ind), int)              # Cell indexes
    for k in range(K):                         # Loop over refinement levels
        ref  = refs[K-k-1]                     # Loop backwards
        pN //= ref                             # Update multipliers
        pM   = 1                               # Initialize multiplier
        for m in range(len(ref)):              # Loop over directions
            Nm   = ind[:,m]//pN[m]             # Coordinate in level
            N   += Nm*np.prod(pN)*pM           # Update N
            ind[:, m] -= Nm*pN[m]              # Update local indexes
            pM  *= ref[m]                      # Update multipliers
    return N

def paramGrid(ind, refs, minX, maxX):          # Arrays of parameters
    pN  = np.prod(refs, axis=0)                # Scale for indexes
    X   = minX + (maxX-minX)*(ind + 0.5)/pN    # Get params
    return X

def refineGrid (N, ref):                       # Refine grid
    siz = np.prod(ref)                         # Size of refined block
    reN = np.arange(siz)                       # Refined indexes
    N   = np.add.outer(reN, N*siz)             # 2D array of indexes
    return N.flatten(order='F')                # Return flattened array

def modelErr(F, ref):
    N   = np.prod(ref)
    pN = 1
    err = []
    for m in range(len(ref)):
        dF  = np.abs(F - np.roll(F, -pN))      # Absolute differences
        dk  = ref[m]*pN                        # Step size
        for n in range(pN):                    # Need pN passes
            dF[dk-pN+n:N:dk] = 0               # Zero wrapped entries
        err.append(dF.max())
        pN *= ref[m]                           # Update counter
    return np.array(err)

def forwardSim(model, X, ref, data):
    times, values, std = data                  # Unpack data tuple
    lnP = np.zeros(len(X))                     # Sum of log(likelihood)
    unc = np.zeros(len(ref))
    for n in range(len(times)):
        F    = model(times[n], X)
        sig  = modelErr(F, ref)
        unc  = np.fmax(unc, sig/F.max())       # Relative sigma for refinement
        sg2  = 2*(sig.max()**2 + std[n]**2)
        lnP -= (F-values[n])**2 / sg2 + np.log(np.pi*sg2)/2
    return lnP, unc

def bayes(model, N, P, refs, minX, maxX, minP, data):        # Driver function
    for nref in range(len(refs)):                            # Loop refinements
        N   = N[np.where(P > minP[nref])]                    # Del P < minP
        N   = refineGrid(N, refs[nref])                      # Refine grid
        Np  = np.prod(refs[nref])                            # Params per set
        P   = np.zeros(len(N))                               # Likelihoods
        unc = np.zeros(len(refs[0]))                         # Uncertainty
        print("ref level, N: ", nref, len(N))
        for n in range(0, len(N), Np):                       # Loop over blks
            Nn  = N[n:n+Np]                                  # Cells block
            ind = indexGrid(Nn,  refs[0:nref+1])             # Get coordinates
            X   = paramGrid(ind, refs[0:nref+1], minX, maxX) # Get params
            Pbk = forwardSim(model, X, refs[nref], data)     # P's for block
            P[n:n+Np] = Pbk[0]
            unc = np.maximum(unc, Pbk[1])
            
        min_lnP = np.amin(P)
        max_lnP = np.amax(P)
        P += (1022*np.log(2) - max_lnP)
        P -= np.log(len(P))
        P = np.exp(P)
        P  /= np.sum(P)                                      # Normalize P's
    return N, P, unc


#-----------------------------------------------------------------------------#


def testData(t, parms):
    f = np.zeros(len(t))
    e = np.ones(len(t))*0.01                          # Uncertainty
    for p in parms[1:]:
        f += np.exp(-p*t)                             # Function
    f *= parms[0]
    return (t, f, e)                                  # Data tuple
def testModel(t, parms):
    f = np.zeros(len(parms))
    for p in parms.T[1:]:
        f += np.exp(-p*t)                             # Function
    f *= parms[:,0]
    return f

import pandas as pd
def get_data(filename):
    # To replace testData
    df = pd.read_hdf(filename)
    t = np.array(df['time'])
    PL = np.array(df['PL'])
    uncertainty = np.array(df['uncertainty'])
    return (t, PL, uncertainty)

def pulse_laser_maxgen(max_gen, alpha, x_array):
    # Initial N, P
    return (max_gen * np.exp(-alpha * x_array))

def simulate_tstep(m, dx, dt, start_t, target_t, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, init_N, init_P, init_E_field):
    current_t = start_t
    
    ## Set initial condition
    y = np.concatenate([init_N, init_P, init_E_field], axis=None)
    
    # Time step until the simulation time matches the next experimental data time
    while (current_t < target_t):
        k1 = dydt(y, m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB)
        k2 = dydt(y + 0.5*dt*k1, m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB)
        k3 = dydt(y + 0.5*dt*k2, m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB)
        k4 = dydt(y + dt*k3, m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB)

        y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        current_t += dt
        
    N = y[0:m]
    P = y[m:2*m]
    E_field = y[2*m:]
    return N, P, E_field

def dydt(y, m, dx, Sf, Sb, mu_n, mu_p, T, n0, p0, tauN, tauP, B, eps, eps0, q, q_C, kB, recycle_photons=True, do_ss=False, alphaCof=0, thetaCof=0, delta_frac=1, fracEmitted=0, combined_weight=0, E_field_ext=0, dEcdz=0, dChidz=0, init_N=0, init_P=0):
    ## Initialize arrays to store intermediate quantities that do not need to be iteratively solved
    # These are calculated at node edges, of which there are m + 1
    # dn/dx and dp/dx are also node edge values
    Jn = np.zeros((m+1))
    Jp = np.zeros((m+1))

    # These are calculated at node centers, of which there are m
    # dE/dt, dn/dt, and dp/dt are also node center values
    dJz = np.zeros((m))
    rad_rec = np.zeros((m))
    non_rad_rec = np.zeros((m))

    N = y[0:m]
    P = y[m:2*(m)]
    E_field = y[2*(m):]
    N_edges = (N[:-1] + np.roll(N, -1)[:-1]) / 2 # Excluding the boundaries; see the following FIXME
    P_edges = (P[:-1] + np.roll(P, -1)[:-1]) / 2
    
    ## Do boundary conditions of Jn, Jp
    # FIXME: Calculate N, P at boundaries?
    Sft = (N[0] * P[0] - n0 * p0) / ((N[0] / Sf) + (P[0] / Sf))
    Sbt = (N[m-1] * P[m-1] - n0 * p0) / ((N[m-1] / Sb) + (P[m-1] / Sb))
    Jn[0] = Sft
    Jn[m] = -Sbt
    Jp[0] = -Sft
    Jp[m] = Sbt

    ## Calculate Jn, Jp [nm^-2 ns^-1] over the space dimension, 
    # Jn(t) ~ N(t) * E_field(t) + (dN/dt)
    # np.roll(y,m) shifts the values of array y by m places, allowing for quick approximation of dy/dx ~ (y[m+1] - y[m-1] / 2*dx) over entire array y
    Jn[1:-1] = (-mu_n * (N_edges) * (q * (E_field[1:-1] + E_field_ext) + dChidz) + 
                (mu_n*kB*T) * ((np.roll(N,-1)[:-1] - N[:-1]) / (dx)))

    ## Changed sign
    Jp[1:-1] = (-mu_p * (P_edges) * (q * (E_field[1:-1] + E_field_ext) + dChidz + dEcdz) -
                (mu_p*kB*T) * ((np.roll(P, -1)[:-1] - P[:-1]) / (dx)))
        
    # [V nm^-1 ns^-1]
    dEdt = (Jn + Jp) * ((q_C) / (eps * eps0))
    
    ## Calculate recombination (consumption) terms
    rad_rec = B * (N * P - n0 * p0)
    non_rad_rec = (N * P - n0 * p0) / ((tauN * P) + (tauP * N))
        
    ## Calculate dJn/dx
    dJz = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / (dx)

    ## N(t) = N(t-1) + dt * (dN/dt)
    dNdt = ((1/q) * dJz - rad_rec - non_rad_rec)
    #if do_ss: dNdt += init_N

    ## Calculate dJp/dx
    dJz = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / (dx)

    ## P(t) = P(t-1) + dt * (dP/dt)
    dPdt = ((1/q) * -dJz - rad_rec - non_rad_rec)
    #if do_ss: dPdt += init_P

    ## Package results
    dydt = np.concatenate([dNdt, dPdt, dEdt], axis=None)
    return dydt

from scipy import integrate as intg
def propagatingPL(dx, N, P, B, n0, p0):
    # Accept a 1D array - the radiative recombination over space at a given time
    # and return the PL integrated over all space at that time
    rad_rec = B * (N * P - n0 * p0)
    PL = intg.trapz(rad_rec, dx=dx)
    
    return PL

if __name__ == "__main__":
    e = get_data("bayesim example expdata.h5")
    ref1 = np.array([4,4,4])
    ref2 = np.array([4,4,4])
    ref3 = np.array([4,4,4])
    refs = [ref1, ref2, ref3]                         # Refinements
    minX = np.array([0, 0, 5])                        # Smallest param values
    maxX = np.array([1, 2,15])                        # Largest param values
    minP = np.array([0, 0.01, 0.05])                  # Threshold P
    N    = np.array([0])                              # Initial N
    P    = np.array([1.0])                            # Initial P
    mP   = ()                                         # Forward model params
    t = np.linspace(0, 5, 100+1)                      # Event tmies
    data = testData(t, np.array([0.7, 1, 10]))        # Test data
    N,P,U  = bayes(testModel, N, P, refs, minX, maxX, minP, data)

    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(N,P)
    
    margins = True
    
    if margins:
        max_P_index = indexGrid(N[np.where(P == np.amax(P))], refs)[0]
        
        for p in range(len(refs[-1])):
            new = np.array([np.ones(max_P_index[p]) * max_P_index[i] if not i == p else np.arange(0, max_P_index[p]) for i in range(len(refs[-1]))], dtype=int).T
            print(p, new)
            cells = indexN(new, refs)
            print(cells)
            
            diff = np.setdiff1d(cells, N)
            if (len(diff)): 
                print("Warning: some cell values not in original N list")
                print(diff)
            
        
        
        # mprob_fig = plt.figure(figsize=(10,10))
        # focused_mprob_fig = plt.figure(figsize=(10,10))
        # cdim = np.ceil(np.sqrt(len(m_probs)))
        # rdim = np.ceil(len(m_probs) / cdim)
                
        # mprob_fig.tight_layout()
        
    Nn  = N[np.where(P > 0.12)]
    ind = indexGrid(Nn,refs)
    X   = minX + ind*(maxX-minX)
    print('Check indexN: ', indexN(ind, refs), Nn)

