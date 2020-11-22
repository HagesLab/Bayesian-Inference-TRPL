 # -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:57:05 2020

@author: cfai2
"""

from matplotlib import pylab as plt
import numpy as np
import scipy.integrate as intg
import scipy.linalg as lin
import time


def excite(a, l, x):
    return (a * np.exp(-x/l))

def Rrad(rate, n, p, n0, p0):
    return rate*(n*p - n0*p0)

def Rnrad(n, p, n0, p0, tauN, tauP):
    return (n*p - n0*p0) / (tauN*p + tauP*n)

def Rsurf(rate, n, p, n0, p0):
    return rate * (n * p - n0 * p0) / (n + p)

def pL(rate, n, p, n0, p0):
    return Rrad(rate, n, p, n0, p0).sum()

def timeStep(P, N, E, N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, scale, a0, a1, a2, MAX=200,TOL=1e-6):
    PO = P[0];  POO = P[1];  P = PO.copy()         # Intitalize densities
    NO = N[0];  NOO = N[1];  N = NO.copy()
    EO = E[0];  EOO = E[1];  E = EO.copy()
    m  = len(P)
    AN = np.zeros((3,m))
    AP = np.zeros((3,m))
    AE = np.ones(m+1)
    bN = np.zeros(m)
    bP = np.zeros(m)
    bE = np.zeros(m+1)
    for iter in range(MAX):                         # Up to MAX iterations
        AN[0,1:]  = DN*(-E[1:-1]/2 - 1)
        AP[0,1:]  = DP*(+E[1:-1]/2 - 1)
        AN[2,:-1] = DN*(+E[1:-1]/2 - 1)
        AP[2,:-1] = DP*(-E[1:-1]/2 - 1)
        AN[1] = a0 - AN[0] - AN[2]
        AP[1] = a0 - AP[0] - AP[2]

    # Recombination terms and RHS vectors
        R  = Rrad(rate, N, P, N0, P0) + Rnrad(N, P, N0, P0, tauN, tauP)
        s0 = Rsurf(sr0, N[0],  P[0],  N0, P0)
        sL = Rsurf(srL, N[-1], P[-1], N0, P0)
        bN = -R - a1*NO - a2*NOO
        bP = -R - a1*PO - a2*POO
        bN[0]  -= s0;  bP[0]  -= s0
        bN[-1] -= sL;  bP[-1] -= sL

    # Linear solve
        NN = N.copy()
        PP = P.copy()
        EE = E.copy()
        N  = lin.solve_banded((1,1), AN, bN)
        P  = lin.solve_banded((1,1), AP, bP)
        AE[1:-1] = scale*( DP*(P[1:]+P[0:-1]) + DN*(N[1:]+N[0:-1]) )/2 + a0
        bE[1:-1] = scale*( DP*(P[1:]-P[0:-1]) - DN*(N[1:]-N[0:-1]) )   - a1*EO[1:-1] - a2*EOO[1:-1]
        E  = bE/AE
        normN = lin.norm(N-NN) / lin.norm(N+TOL)
        normP = lin.norm(P-PP) / lin.norm(P+TOL)
        normE = lin.norm(E-EE) / lin.norm(E+TOL)
        if ((normN< TOL and normP < TOL and normE < TOL)):
    #        print("Converged after {} iterations".format(iter+1))
            return (P, N, E)
    print("FAILED TO CONVERGE after {} iterations".format(iter+1))

def pvsim(Time, Length, L, T, pT, A, l, N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, eps):

    # Non dimensionalize variables
    dx    = Length/L
    dt    = Time/T
    scale = lambda0/dx/eps
    sr0  *= dt/dx
    srL  *= dt/dx
    rate *= dt/dx**3
    tauN /= dt
    tauP /= dt
    N0   *= dx ** 3
    P0   *= dx ** 3
    DN   *= dt/dx**2
    DP   *= dt/dx**2
    A    *= dx**3
    l    /= dx
    params = (N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, scale)

    # Initialization - nodes at 1/2, 3/2 ... M-1/2
    x = np.arange(L) + 0.5
    N = np.zeros((3,L))                       # Include fields at prior steps
    P = np.zeros((3,L))
    E = np.zeros((3,L+1))
    N[0] = excite(A, l, x) + N0
    P[0] = excite(A, l, x) + P0
    pltN = [N[0].copy()/dx**3]                # Initialize plot arrays
    pltP = [P[0].copy()/dx**3]
    pltE = [E[0].copy()/dx]
    pLs  = [pL(rate, N[0], P[0], N0, P0)*dx**4/dt]

    clock0 = time.time()
    for t in range(1,T+1):                    # Main time loop
        if t == 1:                            # Select integration order
            alpha = (1, -1, 0)                # Euler step
        else:
            alpha = (1.5, -2, 0.5)            # 2nd order imPLcit
        N[2] = N[1].copy()                    # Update old densities
        P[2] = P[1].copy()
        E[2] = E[1].copy()
        N[1] = N[0].copy()
        P[1] = P[0].copy()
        E[1] = E[0].copy()
        P[0],N[0],E[0] = timeStep(P[1:], N[1:], E[1:], *params, *alpha)
        pLs.append(pL(rate, N[0], P[0], N0, P0)*dx**4/dt)
        if t%pT == 0:
            pltN.append(N[0].copy()/dx**3)
            pltP.append(P[0].copy()/dx**3)
            pltE.append(E[0].copy()/dx)
    print("Took {} sec".format(time.time() - clock0))

    plt.figure(0)                             # Plots
    plt.clf()
    for t in range(len(pltN)):
        plt.semilogy(x*dx, pltN[t], label="time: {:.1f}".format(t*pT*dt))
    plt.xlim(0,Length)
    plt.xlabel(r'$x [nm]$',      fontsize = 14)
    plt.ylabel(r'$N [nm^{-3}]$', fontsize = 14)
    plt.title('electrons')
    plt.legend()
    plt.figure(1)
    plt.clf()
    for t in range(len(pltP)):
        plt.semilogy(x*dx, pltP[t], label="time: {:.1f}".format(t*pT*dt))
    plt.xlim(0,Length)
    plt.xlabel(r'$x [nm]$',      fontsize = 14)
    plt.ylabel(r'$P [nm^{-3}]$', fontsize = 14)
    plt.title('holes')
    plt.legend()
    plt.figure(2)
    plt.clf()
    for t in range(len(pltN)):
        x = np.arange(L+1)
        plt.plot(x*dx, pltE[t], label="time: {:.1f}".format(t*pT*dt))
    plt.xlim(0,Length)
    plt.xlabel(r'$x [nm]$',      fontsize = 14)
    plt.ylabel(r'$E [nm^{-1}]$', fontsize = 14, labelpad=-3)
    plt.title(r'E field ($\beta qE)$')
    plt.legend()
    plt.figure(3)
    plt.clf()
    t = np.arange(T+1)
    plt.plot(t*dt, pLs)
    plt.xlim(0,Time)
    plt.xlabel(r'$t [ns]$', fontsize = 14)
    plt.ylabel(r'$I\, [nm^{-2} s^{-1}]$',     fontsize = 14)
    plt.title(r'Photo-luminescence intensity')

    return np.array(pLs)


if __name__ == "__main__":

    Time    = 100                             # Final time (ns)
    Length  = 1500                            # Length (nm)
    lambda0 = 704.3                           # q^2/(eps0*k_B T=25C) [nm]
    eps = 13.6                                # dielectric constant
    L   = 150                                 # Spatial points
    T   = 100                                 # Time points
    pT  = 20                                  # Set plot interval
    TOL = 0.00001                             # Convergence tolerance
    MAX = 200                                 # Max iterations

    # Initialization
    A  = 1e-4                                 # Amplitude
    l  = 100                                  # Length scale [nm]

    # Electron/hole density and diffusion
    N0  = 1e-13                               # [/ nm^3]
    P0  = 1e-6                                # [/ nm^3]
    DN  = 2.569257e4                          # [nm^2 / ns]
    DP  = 2.569257e3                          # 1 cm^2/Vs = 2.569257e3 nm^2/ns

    # Recombination rates
    sr0  = 59.50                              # [nm / ns]
    srL  = 1e-8                               # [nm / ns]
    rate = 1e2                                # [nm^3 / ns]
    tauN = 20                                 # [ns]
    tauP = 20                                 # [ns]

    params = (N0, P0, DN, DP, rate, sr0, srL, tauN, tauP, eps)
    pLs    = pvsim(Time, Length, L, T, pT, A, l, *params)

