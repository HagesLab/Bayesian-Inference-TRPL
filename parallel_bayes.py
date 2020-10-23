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
        pN  *= ref                         # Update mutipliers
    return indexes

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

def forwardSim(model, X, init_params, space_grid, ref, data):
    times, values, std = data                  # Unpack data tuple
    lnP = np.zeros(len(X))                     # Sum of log(likelihood)
    # Generate one (the same) initial condition for each X
    spacing = space_grid[1] - space_grid[0]
    m = len(space_grid)
    electron_dens = pulse_laser_maxgen(*(init_params), space_grid)
    electron_dens = np.add.outer(electron_dens, np.zeros(len(X))).T
    hole_dens = pulse_laser_maxgen(*(init_params), space_grid)
    hole_dens = np.add.outer(hole_dens, np.zeros(len(X))).T
    E_field = np.zeros((len(X), m + 1))
    
    electron_dens += X[:,1].reshape(len(X), 1)
    hole_dens += X[:,2].reshape(len(X), 1)
    # X: [B, n0, p0, Sf, Sb, mu_n, mu_p, tau_n, tau_p, T, eps]
    
    # Since simulate() needs info from two time steps at a time, here comes the
    # "first iteration different" design scheme of for loops
    # Either we grab PL of the initial time step and have
    # for n in range(1, len(times)):
        # sim(times[n-1], times[n])
        # PL()
        
    # Or we do
    # for n in range(len(times) - 1):
        # PL()
        # sim(times[n], times[n+1])
    # and grab a final PL after the loop
    
    # I chose the former, for no particular reason
    F = PL(spacing, electron_dens, hole_dens, X[:,0], X[:,1], X[:,2])
    sig  = modelErr(F, ref)
    sg2  = 2*(sig.max()**2 + std[0]**2)
    lnP -= (F-values[0])**2 / sg2 + np.log(np.pi*sg2)/2
    
    for n in range(1, len(times)):
        #F    = model(times[n], X)
        electron_dens, hole_dens, E_field = simulate_tstep(m, spacing, 0.001, times[n-1], times[n], X, 
                                                           electron_dens, hole_dens, E_field)
        
        F = PL(spacing, electron_dens, hole_dens, X[:,0], X[:,1], X[:,2])
        
        sig  = modelErr(F, ref)
        sg2  = 2*(sig.max()**2 + std[n]**2)
        lnP -= (F-values[n])**2 / sg2 + np.log(np.pi*sg2)/2
    return lnP

def marginalP(N, P, refs):                                   # Marginal P's
    pN   = np.prod(refs, axis=0)
    ind  = indexGrid(N, refs)
    marP = []
    for m in range(len(refs[0])):                            # Loop over axes
        Pm   = np.zeros(pN[m])
        for n in np.unique(ind[:,m]):                        # Loop over coord
            Pm[n] = P[np.where(ind[:,m] == n)].sum()         # Marginal P
        marP.append(Pm)                                      # Add to list
    return np.array(marP)

def plotMarginalP(marP, pN, minX, maxX):                     # Plot marginal P
    import matplotlib.pyplot as plt
    plt.clf()
    for m in np.where(pN > 1)[0]:                            # Loop over axes
        plt.figure(m)
        im = np.array([*range(pN[m])]) + 0.5                 # Coords
        X = minX[m] + (maxX[m]-minX[m])*im/pN[m]             # Param values
        plt.plot(X, marP[m], label = str(m))
        plt.xlim(minX[m], maxX[m])
        plt.ylim(0,1)
        plt.legend(loc='best')

def bayes(model, N, P, refs, minX, maxX, init_params, space_grid, minP, data):        # Driver function
    for nref in range(len(refs)):                            # Loop refinements
        N   = N[np.where(P > minP[nref])]                    # P < minP
        N   = refineGrid(N, refs[nref])                      # Refine grid
        Np  = np.prod(refs[nref])                            # Params per set
        lnP = np.zeros(len(N))                               # Likelihoods
        print("ref level, N: ", nref, len(N))
        for n in range(0, len(N), Np):                       # Loop over blks
            Nn  = N[n:n+Np]                                  # Cells block
            ind = indexGrid(Nn,  refs[0:nref+1])             # Get coordinates
            X   = paramGrid(ind, refs[0:nref+1], minX, maxX) # Get params
            Pbk = forwardSim(model, X, init_params, space_grid, refs[nref], data)     # P's for block
            lnP[n:n+Np] = Pbk
        P = np.exp(lnP + 1000*np.log(2) - np.log(len(lnP)) - np.max(lnP))
        P  /= np.sum(P)                                      # Normalize P's
    return N, P


#-----------------------------------------------------------------------------#
import pandas as pd
def get_data(df):
    # To replace testData
    t = np.array(df['time'])
    PL = np.array(df['PL'])
    uncertainty = np.array(df['uncertainty'])
    return (t, PL, uncertainty)

def pulse_laser_maxgen(max_gen, alpha, grid_x):
    # Initial N, P
    return (max_gen * np.exp(-alpha * grid_x))

def simulate_tstep(m, dx, dt, start_t, target_t, X, init_N, init_P, init_E_field):
    # The job of simulate() is to take the N, P, and E-field at the current time and advance them to the next required time
    # X must have a strict order - for here that order is:
    # [B, n0, p0, Sf, Sb, mu_n, mu_p, tau_n, tau_p, T, eps]
    current_t = start_t
    
    ## Set initial condition
    # init_N, init_P, init_E_field = [X][]
    y = np.concatenate([init_N, init_P, init_E_field], axis=1)
    # y = [X][], one N-P-E chain per X
    # Time step until the simulation time matches the next experimental data time
    # FIXME: What happens if the time step takes current_t beyond target_t?
    while (current_t < target_t):
        k1 = dydt(y, m, dx, *(X.T))
        k2 = dydt(y + 0.5*dt*k1, m, dx, *(X.T))
        k3 = dydt(y + 0.5*dt*k2, m, dx, *(X.T))
        k4 = dydt(y + dt*k3, m, dx, *(X.T))

        y = y + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        current_t += dt
        
    N = y[:,0:m]
    P = y[:,m:2*m]
    E_field = y[:,2*m:]
    return N, P, E_field

def dydt(y, m, dx, B, n0, p0, Sf, Sb, mu_n, mu_p, tauN, tauP, T, eps, recycle_photons=True, do_ss=False, alphaCof=0, thetaCof=0, delta_frac=1, fracEmitted=0, combined_weight=0, E_field_ext=0, dEcdz=0, dChidz=0, init_N=0, init_P=0):
    num_sims = len(Sf)
    Jn = np.zeros((num_sims, m+1))
    Jp = np.zeros((num_sims, m+1))

    dJz = np.zeros((num_sims, m))
    rad_rec = np.zeros((num_sims, m))
    non_rad_rec = np.zeros((num_sims, m))

    N = y[:, 0:m]
    P = y[:, m:2*(m)]
    E_field = y[:, 2*(m):]
    N_edges = (N[:,:-1] + np.roll(N, -1, axis=1)[:,:-1]) / 2 # Excluding the boundaries; see the following FIXME
    P_edges = (P[:,:-1] + np.roll(P, -1, axis=1)[:,:-1]) / 2
    
    Sft = (N[:,0] * P[:,0] - n0 * p0) / ((N[:,0] / Sf) + (P[:,0] / Sf))
    Sbt = (N[:,m-1] * P[:,m-1] - n0 * p0) / ((N[:,m-1] / Sb) + (P[:,m-1] / Sb))
    Jn[:,0] = Sft
    Jn[:,m] = -Sbt
    Jp[:,0] = -Sft
    Jp[:,m] = Sbt

    # Params must be promoted to 2D to multiply with 2D N, P, E_field
    Jn[:,1:-1] = (-mu_n.reshape(num_sims, 1) * (N_edges) * (q * (E_field[:,1:-1] + E_field_ext) + dChidz) + 
                (mu_n.reshape(num_sims, 1) * kB * T.reshape(num_sims, 1)) * ((np.roll(N,-1, axis=1)[:,:-1] - N[:,:-1]) / (dx)))

    Jp[:,1:-1] = (-mu_p.reshape(num_sims, 1) * (P_edges) * (q * (E_field[:,1:-1] + E_field_ext) + dChidz + dEcdz) -
                (mu_p.reshape(num_sims, 1) * kB * T.reshape(num_sims, 1)) * ((np.roll(P, -1, axis=1)[:,:-1] - P[:,:-1]) / (dx)))
        
    dEdt = (Jn + Jp) * ((q_C) / (eps.reshape(num_sims, 1) * eps0))
    
    rad_rec = B.reshape(num_sims, 1) * (N * P - n0.reshape(num_sims, 1) * p0.reshape(num_sims, 1))
    non_rad_rec = (N * P - n0.reshape(num_sims, 1) * p0.reshape(num_sims, 1)) / ((tauN.reshape(num_sims, 1) * P) + (tauP.reshape(num_sims, 1) * N))

    dJz = (np.roll(Jn, -1, axis=1)[:,:-1] - Jn[:,:-1]) / (dx)

    dNdt = ((1/q) * dJz - rad_rec - non_rad_rec)

    dJz = (np.roll(Jp, -1, axis=1)[:, :-1] - Jp[:, :-1]) / (dx)

    dPdt = ((1/q) * -dJz - rad_rec - non_rad_rec)

    dydt = np.concatenate([dNdt, dPdt, dEdt], axis=1)
    return dydt

from scipy import integrate as intg
def PL(dx, N, P, B, n0, p0):
    # THis should build off the same strict order as simulate()
    num_sims = len(B)
    rad_rec = B.reshape(num_sims, 1) * (N * P - n0.reshape(num_sims, 1) * p0.reshape(num_sims, 1)) #[][]
    PL = intg.trapz(rad_rec, dx=dx, axis=1) #[]
    
    return PL

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

if __name__ == "__main__":
    # This code follows a strict order of parameters:
    param_names = ["B", "n0", "p0", "Sf", "Sb", "mu_n", "mu_p", "tau_n", "tau_p", "T", "eps"]
    unit_conversions = np.array([((1e7) ** 3) / (1e9), ((1e-7) ** 3), ((1e-7) ** 3), (1e7) / (1e9), (1e7) / (1e9), ((1e7) ** 2) / (1e9), ((1e7) ** 2) / (1e9), 1, 1, 1, 1])
    
    ref1 = np.array([4,1,1,4,1,1,1,1,1,1,1])
    ref2 = np.array([4,1,1,4,1,1,1,1,1,1,1])
    ref3 = np.array([4,1,1,4,1,1,1,1,1,1,1])
    refs = [ref1, ref1, ref1]                         # Refinements
    
    minX = np.array([1e-11, 1e8, 1e15, 1e3, 1e-6, 10, 10, 20, 20, 300, 13.6])                        # Smallest param values
    maxX = np.array([1e-9, 1e8, 1e15, 1e5, 1e-6, 10, 10, 20, 20, 300, 13.6])                        # Largest param values
    
    minP = np.array([0, 0.01, 0.01])                 # Threshold P
    
    # Create space grid
    length = 1500.0
    dx = 10.0
    m = int(0.5 + length / dx)   # Number of space steps ("nodes")
    space_grid = np.linspace(dx / 2,length - dx / 2, m)
    init_params = (1e17 * ((1e-7) ** 3), 1e5 * 1e-7)
    
    N    = np.array([0])                              # Initial N
    P    = np.array([1.0])                            # Initial P
    mP   = ()                                         # Forward model params
    
    experimental_data_filename = "1s bayesim example.h5"
    # experimental_data_filename = "10s bayesim example.h5"
    # experimental_data_filename = "100s bayesim example.h5"
    
    # Pre-checks
    from sys import exit
    try:
        num_params = len(param_names)
        if not (len(unit_conversions) == num_params):
            raise ValueError("Unit conversion array is missing entries")
            
        if not (len(minX) == num_params):
            raise ValueError("Missing min param values")
            
        if not (len(maxX) == num_params):
            raise ValueError("Missing max param values")  
            
        if any(minX <= 0):
            raise ValueError("Invalid param values")
            
        if any(minX > maxX):
            raise ValueError("Min params larger than max params")
            
        # TODO: Additional checks involving refs
            
        print("Starting simulations with the following parameters:")
        for i in range(num_params):
            if minX[i] == maxX[i]:
                print("{}: {}".format(param_names[i], minX[i]))
                
            else:
                print("{}: {} to {}".format(param_names[i], minX[i], maxX[i]))
                
        df = pd.read_hdf(experimental_data_filename)
        print("\nExperimental data - {}".format(experimental_data_filename))
        print(df)
        _continue = input("Continue? (y/n)")
        
        if not (_continue == 'y'): raise KeyboardInterrupt("Aborted")
        
    except Exception as oops:
        print(oops)
        exit(0)
        
    minX *= unit_conversions
    maxX *= unit_conversions
    data = get_data(df)
    N, P = bayes(testModel, N, P, refs, minX, maxX, init_params, space_grid, minP, data)
    marP = marginalP(N, P, refs)
    plotMarginalP(marP, np.prod(refs,axis=0), minX * (unit_conversions ** -1), maxX * (unit_conversions ** -1))




