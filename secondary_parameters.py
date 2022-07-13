# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 12:05:40 2022

@author: cfai2
"""
import numpy as np

def t_rad(B, p0):
    # B [cm^3 / s]
    # p0 [cm^-3]
    return 1 / (B*p0) * 10**9

def t_auger(CP, p0):
    return 1 / (CP*p0**2) * 10**9

def LI_tau_eff(B, p0, tau_n, Sf, Sb, CP, thickness, mu):
    # S [cm/s] -> [nm/ns]
    # B [cm^3 / s]
    # p0 [cm^-3]
    # tau_n [ns]
    # thickness [nm]
    kb = 0.0257 #[ev]
    q = 1
    
    D = mu * kb / q * 10**14 / 10**9# [cm^2 / V s] * [eV] / [eV/V] = [cm^2/s] -> [nm^2/ns]
    tau_surf = (thickness / ((Sf+Sb)*0.01)) + (thickness**2 / (np.pi ** 2 * D))
    t_r = t_rad(B, p0)
    t_aug = t_auger(CP, p0)
    return (t_r**-1 + t_aug**-1 + tau_surf**-1 + tau_n**-1)**-1

def LI_tau_srh(tau_n, Sf, Sb, thickness, mu):
    kb = 0.0257 #[ev]
    q = 1
    
    D = mu * kb / q * 10**14 / 10**9# [cm^2 / V s] * [eV] / [eV/V] = [cm^2/s] -> [nm^2/ns]
    tau_surf = (thickness / ((Sf+Sb)*0.01)) + (thickness**2 / (np.pi ** 2 * D))
    
    return (tau_surf**-1 + tau_n**-1)**-1

def HI_tau_srh(tau_n, tau_p, Sf, Sb, thickness, mu):
    kb = 0.0257 #[ev]
    q = 1
    
    D = mu * kb / q * 10**14 / 10**9# [cm^2 / V s] * [eV] / [eV/V] = [cm^2/s] -> [nm^2/ns]
    tau_surf = 2*(thickness / ((Sf+Sb)*0.01)) + (thickness**2 / (np.pi ** 2 * D))
    
    return (tau_surf**-1 + (tau_n+tau_p)**-1)**-1

def s_eff(sf, sb):
    return sf+sb

def mu_eff(mu_n, mu_p):
    return 2 / (mu_n**-1+mu_p**-1)

def epsilon(lamb):
    return lamb**-1

if __name__ == "__main__":
    print(LI_tau_eff(2.779e16, 4.8e-8, 511, 2, 2, 1e-99, 2000, 20))