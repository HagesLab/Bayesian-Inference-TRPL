#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 00:11:58 2020

@author: tladd
"""

import pickle
import numpy as np
import matplotlib
import tkinter as tk
starting_backend = matplotlib.get_backend()
matplotlib.use("TkAgg")
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
from functools import partial
class TkFrame:

    def __init__(self, title):
        self.outputs_dict = {"N", "P", "E", "PL"}
        self.thr = 0
        
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', False)
        self.root.title(title)
        self.mainframe = tk.Frame(self.root)
        
        self.fig = Figure(figsize=(12,8))
        rdim = cdim = 2
        count = 1
        self.sim_subplots = {}
        for variable in self.outputs_dict:
            self.sim_subplots[variable] = self.fig.add_subplot(rdim, cdim, count)
            self.sim_subplots[variable].set_title(variable)
            count += 1

        self.canvas = tkagg.FigureCanvasTkAgg(self.fig, master=self.mainframe)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0,column=0)
        
        self.fig_toolbar_frame = tk.Frame(master=self.mainframe)
        self.fig_toolbar_frame.grid(row=1,column=0)
        self.fig_toolbar = tkagg.NavigationToolbar2Tk(self.canvas, self.fig_toolbar_frame)

        self.btnframe = tk.Frame(self.mainframe)
        self.btnframe.grid(row=2,column=0)
        
        self.thr_entry = tk.Entry(self.btnframe, width=8)
        self.thr_entry.grid(row=0,column=1)
        self.set_entry(self.thr_entry, 0)
        
        self.increment_thr_btn = tk.Button(self.btnframe, text='>>', command=partial(self.update_thr, +1))
        self.increment_thr_btn.grid(row=0,column=2)
        
        self.decrement_thr_btn = tk.Button(self.btnframe, text='<<', command=partial(self.update_thr, -1))
        self.decrement_thr_btn.grid(row=0,column=0)
        
        self.plot_btn = tk.Button(self.btnframe, text='Plot',command=partial(self.update_thr))
        self.plot_btn.grid(row=1,column=1)
        
        return
    
    def set_entry(self, entry, val):
        entry.delete(0,tk.END)
        entry.insert(0,str(val))
        return
    
    def update_thr(self, increment=None):
        if increment is None:
            self.thr = int(self.thr_entry.get())
            
        else:
            self.thr += increment
        
        if self.thr < 0: self.thr = 0
        if self.thr > len(self.plN) - 1: self.thr = len(self.plN) - 1
        self.set_entry(self.thr_entry, self.thr)
        self.update_plot(self.thr)
        return

    def update_plot(self, pTh=0):
        colors = ['b', 'g', 'r', 'c', 'k', 'm']
        plt = self.sim_subplots["N"]
        plt.cla()
        for t in range(len(self.plN[pTh])):
            plt.semilogy(self.x*self.dx, self.plN[pTh,t], color=colors[t], label="imp: time: {:.1f}".format(self.pT[t]*self.dt))
            plt.semilogy(self.rx*self.rdx, self.refplN[pTh,t], '--', color=colors[t], label="ref: time: {:.1f}".format(self.rpT[t]*self.rdt))
        plt.set_xlim(0,self.Length)
        plt.set_xlabel(r'$x [nm]$',      fontsize = 14)
        plt.set_ylabel(r'$N [nm^{-3}]$', fontsize = 14)
        plt.set_title('electrons')
        plt.legend().set_draggable(True)

        plt = self.sim_subplots["P"]
        plt.cla()
        for t in range(len(self.plP[pTh])):
            plt.semilogy(self.x*self.dx, self.plP[pTh,t], color=colors[t], label="imp: time: {:.1f}".format(self.pT[t]*self.dt))
            plt.semilogy(self.rx*self.rdx, self.refplP[pTh,t], '--', color=colors[t], label="ref: time: {:.1f}".format(self.rpT[t]*self.rdt))
        plt.set_xlim(0,self.Length)
        plt.set_xlabel(r'$x [nm]$',      fontsize = 14)
        plt.set_ylabel(r'$P [nm^{-3}]$', fontsize = 14)
        plt.set_title('holes')
        
        plt = self.sim_subplots["E"]
        plt.cla()
        for t in range(len(self.plE[pTh])):
            x = np.arange(self.L+1)
            rx = np.arange(self.rL + 1)
            plt.plot(x*self.dx, self.plE[pTh,t], color=colors[t], label="imp: time: {:.1f}".format(self.pT[t]*self.dt))
            plt.plot(rx*self.rdx, self.refplE[pTh,t], '--', color=colors[t], label="ref: time: {:.1f}".format(self.rpT[t]*self.rdt))
        plt.set_xlim(0,self.Length)
        plt.set_xlabel(r'$x [nm]$',      fontsize = 14)
        plt.set_ylabel(r'$E [nm^{-1}]$', fontsize = 14, labelpad=-3)
        plt.set_title(r'E field ($\beta qE)$')
        
        plt = self.sim_subplots["PL"]
        plt.cla()
        t = np.arange(self.T//self.plT+1)
        rt = np.arange(self.rT//self.rplT + 1)
        plt.semilogy(t*self.dt*self.plT, self.plI[pTh], label="imp")
        plt.semilogy(rt*self.rdt*self.rplT, self.refplI[pTh], '--', label="ref")
        plt.set_xlim(0,self.Time)
        plt.set_xlabel(r'$t [ns]$', fontsize = 14)
        plt.set_ylabel(r'$I\, [nm^{-2} s^{-1}]$',     fontsize = 14)
        plt.set_title(r'Photo-luminescence intensity')
        if any(self.plI[pTh] < 0) or any(self.refplI[pTh] < 0):
            plt.set_yscale('symlog')
        else:
            plt.set_yscale('log')
            
        self.fig.suptitle("Thread " + str(pTh))
        self.fig.tight_layout()
        self.fig.canvas.draw()
        
        return

    def run(self, data, reference, inputs, ref_inputs):
        self.plN, self.plP, self.plE, self.plI = pickle.load(open(data, 'rb'))
        self.refplN, self.refplP, self.refplE, self.refplI = pickle.load(open(reference,'rb'))

        simPar = pickle.load(open(inputs, 'rb'))[0]
        refsimPar = pickle.load(open(ref_inputs, 'rb'))[0]
        self.Length, self.Time, self.L, self.T, self.plT, self.pT, TOL, MAX = simPar
        self.dx = self.Length/self.L
        self.dt = self.Time/self.T
        self.x  = np.arange(self.L) + 0.5
        
        self.rLength, self.rTime, self.rL, self.rT, self.rplT, self.rpT, rTOL, rMAX = refsimPar
        self.rdx = self.rLength/self.rL
        self.rdt = self.rTime/self.rT
        self.rx  = np.arange(self.rL) + 0.5
        
        self.mainframe.pack(expand=1, fill="both")
        #width, height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()

        #self.root.geometry('%dx%d+0+0' % (width,height))
        self.root.mainloop()
        print("Closed, cleaning up...")
        matplotlib.use(starting_backend)
        return

if __name__ == "__main__":
    f = TkFrame("Plot View")
    
    # pvSim output pickle
    data_path = 'hagesOut700s.pik'
    
    # odeint reference pickle
    ref_path = 'testHagesOut700.pik'
    
    # pvSim output's corresponding input
    input_path = 'hagesInputs700.pik'
    
    # odeint output's corresponding input
    ref_input_path = 'hagesInputs700.pik'
    
    f.run(data_path, ref_path, input_path, ref_input_path)

    