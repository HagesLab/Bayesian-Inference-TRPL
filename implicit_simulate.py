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
q = -1.0 # [e]
q_C = -1.602e-19 # [C]
kB = 8.61773e-5  # [eV / K]
def pulse_laser_maxgen(max_gen, alpha, x_array):
    """

    Parameters
    ----------
    max_gen : [nm^-3]
        Exponential prefactor; value at x=0
    alpha : [nm^-1]
        Exponential decay factor
    x_array : np.ndarray of positional nodes
        

    Returns
    -------
    np.ndarray
        Initial delta_N, delta_P profile
        Amt of excited charge carriers over the standard level

    """
    return (max_gen * np.exp(-alpha * x_array))

def time_step(current_N, current_P, current_E_field, prev_N, prev_P, prev_E_field, dx, dt, D_N, D_P, T, n0, p0, rr_rate, tau_N, tau_P, sf, sb, alpha_0, alpha_1, alpha_2):
    # Do 1st TS using 1st order backward
    new_N = current_N.copy()
    new_P = current_P.copy()
    new_E_field = current_E_field.copy()
    
    
    iter_ = 0
    while(True):
    
        # N, P block
        mat_A_N = np.zeros((m, m))
        mat_A_P = np.zeros((m, m))
        
        mat_A_N[0,0] = alpha_0 * (dx / dt) + D_N * (-q / (2*kB*T) * (new_E_field[1]) - (1/dx))
        mat_A_P[0,0] = alpha_0 * (dx / dt) + D_P * (q / (2*kB*T) * (new_E_field[1]) - (1/dx))
        mat_A_N[0,1] = D_N * (-q / (2*kB*T) * new_E_field[1] + (1/dx))
        mat_A_P[0,1] = D_P * (q / (2*kB*T) * new_E_field[1] + (1/dx))
        
        mat_A_N[m-1,m-1] = alpha_0 * (dx / dt) - D_N * (-q / (2*kB*T) * (new_E_field[m-1]) + (1/dx))
        mat_A_P[m-1,m-1] = alpha_0 * (dx / dt) - D_P * (q / (2*kB*T) * (new_E_field[m-1]) + (1/dx))
        mat_A_N[m-1,m-2] = -D_N * (-q / (2*kB*T) * new_E_field[m-1] - (1/dx))
        mat_A_P[m-1,m-2] = -D_P * (q / (2*kB*T) * new_E_field[m-1] - (1/dx))
        
        # for i in range(1, m-1):
        #     mat_A_N[i,i-1] = -D_N * (-q / (2*kB*T) * new_E_field[i] - (1/dx))
        #     mat_A_P[i,i-1] = -D_P * (q / (2*kB*T) * new_E_field[i] - (1/dx))
            
        #     mat_A_N[i,i] = (dx / dt) + D_N * (-q / (2*kB*T) * (new_E_field[i+1] - new_E_field[i]) - (2/dx))
        #     mat_A_P[i,i] = (dx / dt) + D_P * (q / (2*kB*T) * (new_E_field[i+1] - new_E_field[i]) - (2/dx))
            
        #     mat_A_N[i,i+1] = D_N * (-q / (2*kB*T) * new_E_field[i+1] + (1/dx))
        #     mat_A_P[i,i+1] = D_P * (q / (2*kB*T) * new_E_field[i+1] + (1/dx))
            
        mat_A_N[1:-1,0:-2] += np.diag(-D_N * (-q / (2*kB*T) * new_E_field[1:-2] - (1/dx)), 0)
        mat_A_P[1:-1,0:-2] += np.diag(-D_P * (q / (2*kB*T) * new_E_field[1:-2] - (1/dx)), 0)
        
        mat_A_N[1:-1,1:-1] += np.diag(alpha_0 * (dx / dt) + D_N * (-q / (2*kB*T) * (np.roll(new_E_field, -1)[1:-2] - new_E_field[1:-2]) - (2/dx)), 0)
        mat_A_P[1:-1,1:-1] += np.diag(alpha_0 * (dx / dt) + D_P * (q / (2*kB*T) * (np.roll(new_E_field, -1)[1:-2] - new_E_field[1:-2]) - (2/dx)), 0)
        
        mat_A_N[1:-1,2:] += np.diag(D_N * (-q / (2*kB*T) * np.roll(new_E_field, -1)[1:-2] + (1/dx)))
        mat_A_P[1:-1,2:] += np.diag(D_P * (q / (2*kB*T) * np.roll(new_E_field, -1)[1:-2] + (1/dx)))
        
        rr = -rr_rate * (new_N * new_P - n0 * p0)
        nrr = -(new_N * new_P - n0 * p0) / ((tau_N * new_P) + (tau_P * new_N))
        
        vec_B_N = (alpha_1 * current_N + alpha_2 * prev_N) * (dx/dt) + (rr + nrr) * dx
        vec_B_P = (alpha_1 * current_P + alpha_2 * prev_P) * (dx/dt) + (rr + nrr) * dx
            
        # Boundary consumption terms lag one iteration behind
        f_0 = -sf * (new_N[0] * new_P[0] - n0 * p0) / (new_N[0] + new_P[0])
        f_L = -sb * (new_N[m-1] * new_P[m-1] - n0 * p0) / (new_N[m-1] + new_P[m-1])
        
        vec_B_N[0] += f_0
        vec_B_N[m-1] += f_L
        vec_B_P[0] += f_0
        vec_B_P[m-1] += f_L
        
        new_N = linalg.solve(mat_A_N, vec_B_N)
        new_P = linalg.solve(mat_A_P, vec_B_P)
        
        # E block

        new_E_field[0] = alpha_1 * current_E_field[0] + alpha_2 * prev_E_field[0]
        new_E_field[m] = alpha_1 * current_E_field[m] + alpha_2 * prev_E_field[m]
        
        # for i in range(1,m):
        #     b = E_field[i,k-1] + (q_C * dt / eps) * (D_P/dx * (new_P[i] - new_P[i-1]) - D_N/dx * (new_N[i] - new_N[i-1]))
            
        #     A = 1 - (q_C * dt / eps) * (D_P * (q / (2*kB*T)) * (new_P[i] + new_P[i-1]) + D_N * (q / (2*kB*T)) * (new_N[i] + new_N[i-1]))
        #     new_E_field[i] = (A ** -1) * b
            
        b2 = alpha_1 * current_E_field[1:-1] + alpha_2 * prev_E_field[1:-1] + (q_C * dt / eps) * (D_P/dx * (new_P[1:] - np.roll(new_P, 1)[1:]) - D_N/dx * (new_N[1:] - np.roll(new_N, 1)[1:]))
            
        A2 = alpha_0 - (q_C * dt / eps) * (D_P * (q / (2*kB*T)) * (new_P[1:] + np.roll(new_P, 1)[1:]) + D_N * (q / (2*kB*T)) * (new_N[1:] + np.roll(new_N, 1)[1:]))
        new_E_field[1:-1] = (A2 ** -1) * b2
            
        iter_ += 1
        if (iter_ > 5): break
    
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
    rr_rate = 0*1e-10 * ((1e7) ** 3) / (1e9)# [cm^3 / s] to [nm^3 / ns]
    tau_N = 20000                      # [ns]
    tau_P = 20000                      # [ns]
    
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
    N = np.zeros((m, n+1)) 
    P = np.zeros((m, n+1))  
    E_field = np.zeros((m+1, n+1))
    
    grid_node_x = np.linspace(dx/2,length - dx/2, m)
    # IC at t=0
    N[:, 0] = pulse_laser_maxgen(1e17 * ((1e-7) ** 3), 1e5 * 1e-7, grid_node_x) + n0
    P[:, 0] = pulse_laser_maxgen(1e17 * ((1e-7) ** 3), 1e5 * 1e-7, grid_node_x) + p0


    # Do 1st TS using 1st order backward
        
    iter_ = 0
    startTime = time.time()
    params = (dx, dt, D_N, D_P, T, n0, p0, rr_rate, tau_N, tau_P, sf, sb)
    alphas = (1, 1, 0)
    alphas_2nd = (1.5, 2, -0.5)
    N[:,1], P[:,1], E_field[:,1] = time_step(N[:,0], P[:,0], E_field[:,0], N[:,-1], P[:,-1], E_field[:,-1], *params, *alphas)

    for k in range(2, n+1):
        N[:,k], P[:,k], E_field[:,k] = time_step(N[:,k-1], P[:,k-1], E_field[:,k-1], N[:,k-2], P[:,k-2], E_field[:,k-2], *params, *alphas_2nd)
        
    print("Took {} sec".format(time.time() - startTime))
    #plot the graph
    plt.figure(0)
    plt.yscale('log')
    plt.plot(grid_node_x, N[:,0], label="time: 0.0")
    plt.plot(grid_node_x, N[:,int(final_t*0.1/ dt)], label="time: 0.1")
    plt.plot(grid_node_x, N[:,int(final_t*0.2/ dt)], label="time: 0.2")
    plt.plot(grid_node_x, N[:,int(final_t*0.5/ dt)], label="time: 0.5")
    plt.xlabel('x [nm]', fontsize = 15)
    plt.ylabel('N [nm^-3]', fontsize = 15)
    plt.title('electrons')
    plt.legend()
    
    plt.figure(1)
    plt.yscale('log')
    plt.plot(grid_node_x, P[:,0], label="time: 0.0")
    plt.plot(grid_node_x, P[:,int(final_t*0.1/ dt)], label="time: 0.1")
    plt.plot(grid_node_x, P[:,int(final_t*0.2/ dt)], label="time: 0.2")
    plt.plot(grid_node_x, P[:,int(final_t*0.5/ dt)], label="time: 0.5")
    plt.xlabel('x [nm]', fontsize = 15)
    plt.ylabel('P [nm^-3]', fontsize = 15)
    plt.title('holes')
    plt.legend()
