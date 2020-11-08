# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:57:05 2020

@author: cfai2
"""

from matplotlib import pylab as plt
import numpy as np
from scipy import linalg
import time

eps0 = 8.854 * 1e-12 * 1e-9 # [C / V m] to {C / V nm}
q = 1.0 # [e]
q_C = 1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]

TOL = 0.001
MAX_ITER = 100

def pulse_laser_maxgen(max_gen, alpha, x_array):
    return (max_gen * np.exp(-alpha * x_array))

def rr(rate, n, p, n0, p0):
    return rate * (n * p - n0 * p0)

def nrr(n, p, n0, p0, tau_N, tau_P):
    return (n * p - n0 * p0) / ((tau_N * p) + (tau_P * n))

def surface_consumption(rate, n, p, n0, p0):
    return rate * (n * p - n0 * p0) / (n + p)

def time_step(current_N, current_P, current_E_field, prev_N, prev_P, prev_E_field, lamb, D_N, D_P, n0, p0, rr_rate, tau_N, tau_P, sf, sb, alpha_0, alpha_1, alpha_2):
    # Do 1st TS using 1st order backward
    new_N = current_N.copy()
    new_P = current_P.copy()
    new_E_field = current_E_field.copy()

    iter_ = 0
    while(True):
    
        # N, P block
        mat_A_N = np.zeros((3,m))
        mat_A_P = np.zeros((3,m))
        mat_A_N[1,0] = alpha_0 + D_N * (-1/2 * new_E_field[1] + 1)
        mat_A_P[1,0] = alpha_0 + D_P * (1/2 * new_E_field[1] + 1)
        mat_A_N[0,1] = D_N * (-1/2 * new_E_field[1] - 1)
        mat_A_P[0,1] = D_P * (1/2 * new_E_field[1] - 1)
        
        mat_A_N[1,-1] = alpha_0 + D_N * (1/2 * new_E_field[m-1] + 1)
        mat_A_P[1,-1] = alpha_0 + D_P * (-1/2 * new_E_field[m-1] + 1)
        mat_A_N[2,-2] = D_N * (1/2 * new_E_field[m-1] - 1)
        mat_A_P[2,-2] = D_P * (-1/2* new_E_field[m-1] - 1)
        
        mat_A_N[2,0:-2] = D_N * (1/2* new_E_field[1:-2] - 1)
        mat_A_P[2,0:-2] = D_P * (-1/2 * new_E_field[1:-2] - 1)
        
        mat_A_N[1,1:-1] = alpha_0 + D_N * (-1/2 * (np.roll(new_E_field, -1)[1:-2] - new_E_field[1:-2]) + 2)
        mat_A_P[1,1:-1] = alpha_0 + D_P * (1/2 * (np.roll(new_E_field, -1)[1:-2] - new_E_field[1:-2]) + 2)
        
        mat_A_N[0,2:] = D_N * (-1/2 * np.roll(new_E_field, -1)[1:-2] - 1)
        mat_A_P[0,2:] = D_P * (1/2 * np.roll(new_E_field, -1)[1:-2] - 1)
        
        # Volume and surface consumption terms lag one iteration behind
        consumption = rr(rr_rate, new_N, new_P, n0, p0) + nrr(new_N,new_P,n0,p0,tau_N,tau_P)
        
        vec_B_N = (alpha_1 * current_N + alpha_2 * prev_N) - consumption
        vec_B_P = (alpha_1 * current_P + alpha_2 * prev_P) - consumption
            
        f_0 = surface_consumption(sf, new_N[0], new_P[0], n0, p0)
        f_L = surface_consumption(sb, new_N[m-1], new_P[m-1], n0, p0)
        
        vec_B_N[0] -= f_0
        vec_B_N[m-1] -= f_L
        vec_B_P[0] -= f_0
        vec_B_P[m-1] -= f_L
        
        old_P = new_P.copy()
        old_N = new_N.copy()
        new_N = linalg.solve_banded((1,1), mat_A_N, vec_B_N)
        new_P = linalg.solve_banded((1,1), mat_A_P, vec_B_P)
        
        # E block
        new_E_field[0] = alpha_1 * current_E_field[0] + alpha_2 * prev_E_field[0]
        new_E_field[m] = alpha_1 * current_E_field[m] + alpha_2 * prev_E_field[m]
            
        b2 = alpha_1 * current_E_field[1:-1] + alpha_2 * prev_E_field[1:-1] + lamb * (D_P * (new_P[1:] - np.roll(new_P, 1)[1:]) - D_N * (new_N[1:] - np.roll(new_N, 1)[1:]))
            
        A2 = alpha_0 + (lamb/2) * (D_P * (new_P[1:] + np.roll(new_P, 1)[1:]) + D_N * (new_N[1:] + np.roll(new_N, 1)[1:]))
        new_E_field[1:-1] = (A2 ** -1) * b2
            
        iter_ += 1
        norm_diff_p = np.linalg.norm(old_P - new_P) / np.linalg.norm(new_P)
        norm_diff_n = np.linalg.norm(old_N - new_N) / np.linalg.norm(new_N)
        if ((norm_diff_n < TOL and norm_diff_p < TOL) or iter_ > MAX_ITER): 
            #print("Took {} iterations".format(iter_))
            break
    
    return new_N, new_P, new_E_field

if __name__ == "__main__":
    dx = 10
    dt = 0.05
    
    final_t = 100
    length = 1500
    
    # Diffusivity
    mu_N = 10 * ((1e7) ** 2) / (1e9)# [cm^2 / V s] to [nm^2 / V ns]
    mu_P = 10 * ((1e7) ** 2) / (1e9)# [cm^2 / V s] to [nm^2 / V ns]
    
    # Consumption rates
    sf = 5950 * (1e7) / (1e9)        # [cm / s] to [nm / ns]
    sb = 1e-6 * (1e7) / (1e9)       # [cm / s] to [nm / ns]
    rr_rate = 1e-10 * ((1e7) ** 3) / (1e9)# [cm^3 / s] to [nm^3 / ns]
    tau_N = 20                      # [ns]
    tau_P = 20                      # [ns]
    
    # Other
    T = 300               # [K]
    n0 = 1e8 * ((1e-7) ** 3)        # [cm^-3] to [nm^-3]
    p0 = 1e15 * ((1e-7) ** 3)       # [cm^-3] to [nm^-3]
    eps = eps0 * 13.6
    
    D_N = mu_N * kB * T / q
    D_P = mu_P * kB * T / q
    m = int(0.5 + length / dx)
    n = int(0.5 + final_t / dt)
    
    # Node centers placed at x=dx/2 and x=length-dx/2
    N = np.zeros((m, 3)) 
    P = np.zeros((m, 3))  
    E_field = np.zeros((m+1, 3))
    
    grid_node_x = np.linspace(dx/2,length - dx/2, m)
    # IC at t=0
    N[:, 0] = pulse_laser_maxgen(1e17 * ((1e-7) ** 3), 1e5 * 1e-7, grid_node_x) + n0
    P[:, 0] = pulse_laser_maxgen(1e17 * ((1e-7) ** 3), 1e5 * 1e-7, grid_node_x) + p0

    # Non dimensionalize
    n0 *= dx ** 3
    p0 *= dx ** 3
    N[:,0] *= dx ** 3
    P[:,0] *= dx ** 3
    
    D_N *= (dt * dx ** -2)
    D_P *= (dt * dx ** -2)
    
    sf *= (dt * dx ** -1)
    sb *= (dt * dx ** -1)
    rr_rate *= (dt * dx ** -3)
    
    tau_N *= dt ** -1
    tau_P *= dt ** -1
    lamb = (q * q_C) / (eps * kB * T * dx)
    # Do 1st TS using 1st order backward
        
    startTime = time.time()
    params = (lamb, D_N, D_P, n0, p0, rr_rate, tau_N, tau_P, sf, sb)
    alphas = (1, 1, 0)
    alphas_2nd = (1.5, 2, -0.5)
    
    plot_N = N[:,0].copy().reshape((m, 1))
    plot_P = P[:,0].copy().reshape((m, 1))
    plot_tsteps = np.linspace(0, n, 6)
    
    N[:,1], P[:,1], E_field[:,1] = time_step(N[:,0], P[:,0], E_field[:,0], N[:,-1], P[:,-1], E_field[:,-1], *params, *alphas)
    
    for k in range(2, n+1):
        # Circular loop over current and two prev timesteps
        i = k % 3
        N[:, i], P[:,i], E_field[:, i] = time_step(N[:,i-1], P[:,i-1], E_field[:,i-1], N[:,i-2], P[:,i-2], E_field[:,i-2], *params, *alphas_2nd)
    
        if k in plot_tsteps:
            plot_N = np.concatenate((plot_N, N[:,i].reshape((m,1))), axis=1)
            plot_P = np.concatenate((plot_P, P[:,i].reshape((m,1))), axis=1)
    print("Took {} sec".format(time.time() - startTime))
    
    #plot
    plt.figure(0)
    plt.yscale('log')
    t_frac=0
    for N in plot_N.T:
        plt.plot(grid_node_x, N, label="time: {:.2f}".format(plot_tsteps[t_frac] / n * final_t))
        t_frac += 1
    plt.xlabel('x [nm]', fontsize = 15)
    plt.ylabel('N* [unitless]', fontsize = 15)
    plt.title('electrons')
    plt.legend()
    
    plt.figure(1)
    plt.yscale('log')
    t_frac=0
    for P in plot_P.T:
        plt.plot(grid_node_x, P, label="time: {:.2f}".format(plot_tsteps[t_frac] / n * final_t))
        t_frac += 1
    plt.xlabel('x [nm]', fontsize = 15)
    plt.ylabel('P* [unitless]', fontsize = 15)
    plt.title('holes')
    plt.legend()
