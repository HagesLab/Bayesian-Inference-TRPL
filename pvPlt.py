#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 00:11:58 2020

@author: tladd
"""

pTh = 1

import pickle
import numpy as np
import matplotlib.pyplot as plt

#plN, plP, plE, plI = pickle.load(open('pvOut.pik', 'rb'))
plN, plP, plE, plI = pickle.load(open('hagesOut.pik', 'rb'))
sn = np.where((plN < 0).any(axis=2).any(axis=1), 1, 0)
sp = np.where((plP < 0).any(axis=2).any(axis=1), 1, 0)
print(np.sum(sn))
print(np.sum(sp))

simPar = pickle.load(open('hagesInputs.pik', 'rb'))[0]
Length, Time, L, T, plT, pT, TOL, MAX = simPar
dx = Length/L
dt = Time/T
x  = np.arange(L) + 0.5

plt.figure(0)                             # Plot data for thread pTh
plt.clf()
for t in range(len(plN[pTh])):
    plt.semilogy(x*dx, plN[pTh,t], label="time: {:.1f}".format(pT[t]*dt))
plt.xlim(0,Length)
plt.xlabel(r'$x [nm]$',      fontsize = 14)
plt.ylabel(r'$N [nm^{-3}]$', fontsize = 14)
plt.title('electrons')
plt.legend()
plt.figure(1)
plt.clf()
for t in range(len(plP[pTh])):
    plt.semilogy(x*dx, plP[pTh,t], label="time: {:.1f}".format(pT[t]*dt))
plt.xlim(0,Length)
plt.xlabel(r'$x [nm]$',      fontsize = 14)
plt.ylabel(r'$P [nm^{-3}]$', fontsize = 14)
plt.title('holes')
plt.legend()
plt.figure(2)
plt.clf()
for t in range(len(plE[pTh])):
    x = np.arange(L+1)
    plt.plot(x*dx, plE[pTh,t], label="time: {:.1f}".format(pT[t]*dt))
plt.xlim(0,Length)
plt.xlabel(r'$x [nm]$',      fontsize = 14)
plt.ylabel(r'$E [nm^{-1}]$', fontsize = 14, labelpad=-3)
plt.title(r'E field ($\beta qE)$')
plt.legend()
plt.figure(3)
plt.clf()
t = np.arange(T//plT+1)
plt.plot(t*dt*plT, plI[pTh])
plt.xlim(0,Time)
plt.xlabel(r'$t [ns]$', fontsize = 14)
plt.ylabel(r'$I\, [nm^{-2} s^{-1}]$',     fontsize = 14)
plt.title(r'Photo-luminescence intensity')