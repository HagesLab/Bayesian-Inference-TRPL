# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:59:45 2022

@author: cfai2
"""
import numpy as np
from scipy.integrate import solve_ivp, simpson
import time

## Define constants
eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kBT = .02569257  # [eV]
lambda0 = 704.3

def dydt2(t, y, m, dx, Sf, Sb, mu_n, mu_p, n0, p0, CN, CP, tauN, tauP, B, 
          eps):
    """Derivative function for drift-diffusion-decay carrier model."""
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
    Sft = Sf * (N[0] * P[0] - n0 * p0) / (N[0] + P[0])
    Sbt = Sb * (N[m-1] * P[m-1] - n0 * p0) / (N[m-1] + P[m-1])
    Jn[0] = Sft
    Jn[m] = -Sbt
    Jp[0] = -Sft
    Jp[m] = Sbt

    ## Calculate Jn, Jp [nm^-2 ns^-1] over the space dimension, 
    # Jn(t) ~ N(t) * E_field(t) + (dN/dt)
    # np.roll(y,m) shifts the values of array y by m places, allowing for quick approximation of dy/dx ~ (y[m+1] - y[m-1] / 2*dx) over entire array y
    Jn[1:-1] = (mu_n * (N_edges) * (q * E_field[1:-1]) 
                + (mu_n*kBT) * ((np.roll(N,-1)[:-1] - N[:-1]) / (dx)))

    ## Changed sign
    Jp[1:-1] = (mu_p * (P_edges) * (q * E_field[1:-1]) 
                - (mu_p*kBT) * ((np.roll(P, -1)[:-1] - P[:-1]) / (dx)))

    # [V nm^-1 ns^-1]
    dEdt = -(Jn + Jp) * ((q_C) / (eps * eps0))
    ## Calculate recombination (consumption) terms
    rad_rec = B * (N * P - n0 * p0)
    non_rad_rec = (N * P - n0 * p0) / ((tauN * P) + (tauP * N))
    
    auger = (CN * N + CP * P) * (N * P - n0 * p0) 
    ## Calculate dJn/dx
    dJz = (np.roll(Jn, -1)[:-1] - Jn[:-1]) / (dx)

    ## N(t) = N(t-1) + dt * (dN/dt) roughly
    dNdt = ((1/q) * dJz - rad_rec - non_rad_rec - auger)

    ## Calculate dJp/dx
    dJz = (np.roll(Jp, -1)[:-1] - Jp[:-1]) / (dx)

    ## P(t) = P(t-1) + dt * (dP/dt) roughly
    dPdt = ((-1/q) * dJz - rad_rec - non_rad_rec - auger)

    ## Package results
    dydt = np.concatenate([dNdt, dPdt, dEdt], axis=None)
    return dydt

def pvSim_cpu_fallback(plI, matPar, simPar, init_dN):
    """ CPU version of pvSimPCR absorber simulation """
    
    Length, Time, L, T, plT, pT, tol, MAX = simPar
    dx = Length/L
    clock0 = time.perf_counter()
    
    for i, mp in enumerate(matPar):
        n0, p0, DN, DP, B, Sf, Sb, CN, CP, tauN, tauP, lambda_, mag_offset = mp
        mu_n = DN / (kBT)
        mu_p = DP / (kBT)
        args=(L, dx, Sf, Sb, mu_n, mu_p, n0, p0, CN, CP, 
                tauN, tauP, B, (lambda_/lambda0)**-1)

        teff = LI_tau_eff(B, p0, tauN, Sf, Sb, CP, Length, mu_n)
        if teff < Time / 100:
            hmax = 0.025
        else:
    	    hmax = 1
        
        init_N = init_dN + n0
        init_P = init_dN + p0
        init_E = np.zeros(len(init_N) + 1)
        init_condition = np.concatenate([init_N, init_P, init_E])
        ## Do n time steps
        tSteps = np.linspace(0, Time, T+1)
        
        sol = solve_ivp(dydt2, [0,Time], init_condition, args=args, t_eval=tSteps, method='BDF', max_step=1, rtol=1e-5, atol=1e-8)
        
        data = sol.y
        N = data[0:L]
        P = data[L:2*L]
        PL = simpson(B * (N*P - n0*p0), dx=dx, axis=0)
        plI[i] = PL
    
    solver_time = time.perf_counter() - clock0
    
    return solver_time

def t_rad(B, p0):
    # B [nm^3 / ns]
    # p0 [nm^-3]
    if B == 0 or p0 == 0:
        return np.inf
    
    else:
        return 1 / (B*p0)

def t_auger(CP, p0):
    if CP == 0 or p0 == 0:
        return np.inf
    
    else:
        return 1 / (CP*p0**2)

def LI_tau_eff(B, p0, tau_n, Sf, Sb, CP, thickness, mu):
    # S [nm/ns]
    # B [nm^3 / ns]
    # p0 [nm^-3]
    # tau_n [ns]
    # thickness [nm]
    kb = 0.0257 #[ev]
    q = 1
    
    D = mu * kb / q # [nm^2/ns]
    if Sf+Sb == 0 or D == 0:
        tau_surf = np.inf
    else:
        tau_surf = (thickness / ((Sf+Sb))) + (thickness**2 / (np.pi ** 2 * D))
    t_r = t_rad(B, p0)
    t_aug = t_auger(CP, p0)
    return (t_r**-1 + t_aug**-1 + tau_surf**-1 + tau_n**-1)**-1
