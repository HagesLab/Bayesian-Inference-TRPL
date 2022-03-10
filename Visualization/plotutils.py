# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 12:47:23 2022

@author: cfai2
"""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from seaborn import heatmap
import tkinter as tk
import logging

class PlotState():
    
    def __init__(self):
        return
    
    def fromTK_get_axis_limits(self, data, param_limits: dict):
        self.axis_limits = {}
        for param, limit in param_limits.items():
            if data.SECONDARY_PARAMS[param]: continue
            self.axis_limits[param] = (float(limit[0].get()), float(limit[1].get()))
            
    def update_secondary_axis_limits(self, data):
        for sp in data.SECONDARY_PARAMS:
            if sp in self.enabled_params:
                self.axis_limits[sp] = (np.amin(data.X[sp]), 
                                        np.amax(data.X[sp]))
                
                if self.axis_limits[sp][0] == self.axis_limits[sp][1]:
                    self.axis_limits[sp] = (self.axis_limits[sp][0] / 2,
                                            self.axis_limits[sp][1] * 2)
        
    def generate_P_ranking(self, data, num_maxes):
        self.p_rank = np.argsort(data.LL)[::-1]
        self.p_rank = self.p_rank[:num_maxes]
        
    def add_mark_vals(self, marked_values: list):
        self.marked_vals = {}
        for param in self.enabled_params:
            val = marked_values[param].get()
            if not val: continue
        
            try:
                val = float(val)
            except Exception:
                logging.warning(f"invalid mark value for param {param}")
                continue
            
            if not (self.axis_limits[param][0] < val < self.axis_limits[param][1]):
                logging.warning(f"Mark value for {param} exceeds limits")
            
            self.marked_vals[param] = val
            
    def loglimits(self):
        for param, do_log in self.do_logscale.items():
            if do_log:
                self.axis_limits[param] = (np.log10(self.axis_limits[param][0]), np.log10(self.axis_limits[param][1]))
                if param in self.marked_vals: self.marked_vals[param] = np.log10(self.marked_vals[param])
                
    def setup_2D(self, num_plots):
        
        self.fig = Figure(figsize=(7,7), dpi=120)
        self.subplots = [[None for n in range(num_plots)] for nn in range(num_plots)]
        for i in range(num_plots):
            for j in range(num_plots):
                if i >= j:
                    self.subplots[i][j] = self.fig.add_subplot(num_plots, num_plots, (j+1)+i*num_plots)
    
        return
    
    def setup_cov(self):
        self.cov_fig = Figure(figsize=(7,7))
        self.cov_ax = self.cov_fig.add_subplot(1,1,1)
        return
    
    def heatmap_cov(self, cov):
        heatmap(cov, xticklabels=self.enabled_params, yticklabels=self.enabled_params, 
                annot=True, fmt=".3g",ax=self.cov_ax, cmap='coolwarm')
        
        self.cov_fig.canvas.draw()
        return
    
    def make_2D_grid(self, data):
        for i, py in enumerate(self.enabled_params):
            for j, px in enumerate(self.enabled_params):
                if i < j:
                    continue
    
                elif i == j:
                    self.make_1D_plot(data, i, j, py)
                    
                else:
                    self.make_2D_plot(data, i, j, px, py)
                    
                            
        self.fig.tight_layout(pad=0.1)
        self.fig.canvas.draw()
    
    def make_1D_plot(self, data, i, j, param):
        self.subplots[i][j].set_title("{} [{}]".format(param, self.units[param]))
        self.subplots[i][j].set_ylabel("P. Density")
        self.subplots[i][j].set_yscale('linear')
        
        self.subplots[i][j].set_xlim(self.axis_limits[param])
        if param in self.marked_vals: self.subplots[i][j].axvline(self.marked_vals[param], color='r',linewidth=1)
        
        bins = data.h_1D[param][1]
        self.subplots[i][j].bar(bins[:-1], data.h_1D[param][0], width=np.diff(bins), align='edge')
        
        for xb in bins:
            self.subplots[i][j].axvline(xb, color=(0.1,0.1,0.1,0.2), linewidth=0.2)

        if self.do_logscale[param]:
            make_logticks(self.subplots[i][j], self.axis_limits[param], axis='x', override="S_F+S_B" in param)
            
    def make_2D_plot(self, data, i, j, px, py):
        marP_corr, X_corr, Y_corr = data.h_2D[(px, py)] 
        im_corr = self.subplots[i][j].pcolormesh(Y_corr, X_corr, marP_corr.T, cmap='Blues')
        #self.fig.colorbar(im_corr, ax=self.subplots[i][j])
        
        
        self.subplots[i][j].set_xlabel("{} [{}]".format(px, self.units[px]))
        self.subplots[i][j].set_ylabel("{} [{}]".format(py, self.units[py]))

        self.subplots[i][j].scatter(data.X[px][self.p_rank], data.X[py][self.p_rank], s=12, c='gray', marker='o')
                
        if px in self.marked_vals and py in self.marked_vals: 
            self.subplots[i][j].scatter(self.marked_vals[px], self.marked_vals[py], s=20, c='r', marker='o')

        if self.do_logscale[px]:
            make_logticks(self.subplots[i][j], self.axis_limits[px], axis='x', override="S_F+S_B" in px)
            
        if self.do_logscale[py]:
            make_logticks(self.subplots[i][j], self.axis_limits[py], axis='y', override="S_F+S_B" in py)

            
def make_logticks(subplot, axlim, axis='y', override=False):
    """Format ticks as "10^X" for log-scaled plots.

    Parameters
    ----------
    subplot : matplotlib subplot
        
    axlim : 2-tuple
        Axis limits.
    axis : char, optional
        Which axis. The default is 'y'.

    Returns
    -------
    None.

    """
    if axis=='x':
        sx = subplot.set_xticks
        sxl = subplot.set_xticklabels
    else:
        sx = subplot.set_yticks
        sxl = subplot.set_yticklabels
        
    xt = subplot.get_yticks() if axis=='y' else subplot.get_xticks()
    
    #spacing = 1
    #proposed_xt = np.arange(min(xt), max(xt)+spacing, spacing)
    proposed_xt = [tick for tick in xt if int(tick) == tick 
                    and tick < axlim[1] 
                    and tick > axlim[0]]
    if len(proposed_xt) == 0:
        proposed_xt = [tick for tick in xt if int(tick) == tick]
    # while len(proposed_xt) > 5:
    #     spacing += 1
    #     proposed_xt = np.arange(min(xt), max(xt)+spacing, spacing)
        
    xt = proposed_xt
    
    xt.insert(0, xt[0]-1)
    xt.append(xt[-1]+1)
    
    if override:
        xt = [-4,-3,-2,-1,0,1,2,3,4]
    
    sx(xt)
    sxl([r"10$^{{{}}}$".format(int(tick)) for tick in xt])
    
    if override:
        xt = [-4,-3,-2,-1,0,1,2,3,4]
        sxl([r"10$^{-4}$", "", "", "", r"10$^{0}$", "", "", "", r"10$^{4}$"])
    xt = np.arange(min(xt), max(xt)+1, 1)
    sx(np.concatenate([i + np.log10(np.linspace(1,10,10)) for i in xt[:-1]]),minor=True)
    return