# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 12:14:39 2022

@author: cfai2
"""
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

from secondary_parameters import t_rad, LI_tau_eff, mu_eff, s_eff, epsilon
from multiprocessing import Pool
from functools import partial
from scipy.optimize import fmin

class LikelihoodData():
    
    def __init__(self):
        return
    
    def load(self, ll_fname):
        dname = os.path.dirname(ll_fname)
        bname = os.path.basename(ll_fname)
        bname = bname[:bname.find("_BAYRAN_")]
        
        self.X = np.load(os.path.join(dname, f"{bname}_BAYRAN_X.npy"))
        self.LL = np.load(os.path.join(dname, f"{bname}_BAYRAN_P.npy"))
        return
    
    def pack_X_param_indexable(self):
        X = {}
        for i, param in enumerate(self.PARAM_ORDER):
            if not self.SECONDARY_PARAMS[param]:
                X[param] = np.array(self.X[:,i])
                
        self.X = dict(X)
        
    def exclude_using_axis_limits(self, plots):
        exclusion_limits = {param:plots.axis_limits[param] for param in plots.enabled_params if not self.SECONDARY_PARAMS[param]}
        where_in_bins, self.X = exclude(self.X, exclusion_limits, self.PARAM_ORDER)
        self.LL = self.LL[where_in_bins]
        print("Kept {} of {}".format(len(self.LL), len(where_in_bins)))
        
    def calculate_secondary_params(self, enabled_params: list):
        if r"$\mu\prime$" in enabled_params:
            mu_total = mu_eff(self.X[r"$\mu_n$"], self.X[r"$\mu_p$"])
            self.X[r"$\mu\prime$"] = mu_total
            
        if r"$\tau_{eff}$" in enabled_params:
            mu_total = mu_eff(self.X[r"$\mu_n$"], self.X[r"$\mu_p$"])
            tau_eff = LI_tau_eff(self.X[r"$k^*$"], self.X[r"$p_0$"], self.X[r"$\tau_n$"], 
                                 self.X[r"$S_F$"], self.X[r"$S_B$"], self.thickness, mu_total)
            self.X[r"$\tau_{eff}$"] = tau_eff
            
        if r"$\tau_{rad}$" in enabled_params:
            t_r = t_rad(self.X[r"$k^*$"], self.X[r"$p_0$"])
            self.X[r"$\tau_{rad}$"] = t_r
            
        if r"$(S_F+S_B)$" in enabled_params:
            s_total = s_eff(self.X[r"$S_F$"], self.X[r"$S_B$"])
            self.X[r"$(S_F+S_B)$"] = s_total
            
        if r"$\epsilon$" in enabled_params:
            eps = epsilon(self.X[r"$\lambda$"])
            self.X[r"$\epsilon$"] = eps
            
        if r"$\tau_n+\tau_p$" in enabled_params:
            t_sum = self.X[r"$\tau_n$"] + self.X[r"$\tau_p$"]
            self.X[r"$\tau_n+\tau_p$"] = t_sum
        
    def strip_unneeded_params(self, enabled_params: list):
        for param in list(self.X.keys()):
            if not param in enabled_params:
                self.X.pop(param)
                
    def logX(self, plots):
        for param, do_log in plots.do_logscale.items():
            if do_log:
                self.X[param] = np.log10(self.X[param])
        
    def marginalize1D(self, plots):
        X0 = [self.X[p] for p in plots.enabled_params]
        
        with Pool(min(len(plots.enabled_params), os.cpu_count())) as pool:
            g = pool.starmap(partial(marginalize_1D, self.P, plots.axis_limits, plots.bin_count, self.SECONDARY_PARAMS), zip(plots.enabled_params, X0))
        self.h_1D = {plots.enabled_params[i]:g[i] for i in range(len(plots.enabled_params))}
        # self.h_1D = {}
        # for param in plots.enabled_params:
        #     self.h_1D[param] = marginalize_1D(P, axis_overrides, bin_count, self.data.SECONDARY_PARAMS, param, X[param])
        
    def marginalize2D(self, plots):
        param_pairs = []
        for i, py in enumerate(plots.enabled_params):
            for j, px in enumerate(plots.enabled_params):
                if i > j:
                    param_pairs.append((px, py))
                    
        if len(param_pairs) > 0:
            X0 = [self.X[p[0]] for p in param_pairs]
            X1 = [self.X[p[1]] for p in param_pairs]
            with Pool(min(len(param_pairs), os.cpu_count())) as pool:
                g = pool.starmap(partial(marginalize_2D, self.P, plots.axis_limits, plots.bin_count, self.SECONDARY_PARAMS), zip(param_pairs, X0, X1))
                        
            self.h_2D = {param_pairs[i]:g[i] for i in range(len(param_pairs))}
        # self.plots.h_2D = {}
        # for pp in param_pairs:
        #     self.plots.h_2D[pp] = marginalize_2D(P, axis_overrides, bin_count, self.data.SECONDARY_PARAMS, pp, X[pp[0]], X[pp[1]])
          
    def stats_summarize(self):
        means_and_stds = {}
        for param in self.X:
            means_and_stds[param] = (w_mean(self.X[param], self.P),
                                     w_sample_var(self.X[param], self.P, np.sum(self.P**2)), 
                                     np.sum(self.P**2))
            
        return means_and_stds
        
    def calc_max_uncertainty(self):
        uncertainty = {}
        for param in self.X:
            uncertainty[param] = find_best_tf(self.X[param], self.LL, 
                                              self.num_observations / 2000)
        return uncertainty
    
    def calc_covariance(self, plots):
        cov = np.zeros((len(plots.enabled_params), len(plots.enabled_params)))
        for i, param1 in enumerate(plots.enabled_params):
            for j, param2 in enumerate(plots.enabled_params):
                if i > j: 
                    cov[i,j] = cov[j,i]
                else:
                    cov[i,j] = covariance(self.X[param1], self.X[param2], self.P)
        return cov
    
def exclude(arr, limits, param_order):
    where_to_exclude = np.ones(len(arr), dtype=bool)
    for param in limits:
        where_param = param_order.index(param)
        a = np.logical_and(arr[:, where_param] <= limits[param][1], 
                           arr[:, where_param] >= limits[param][0])
            
        where_to_exclude = np.logical_and(a, where_to_exclude)
        
    arr = arr[where_to_exclude]
    return where_to_exclude, arr

def normalize(lnP):
    # Normalization scheme - to ensure that np.sum(P) is never zero due to mass underflow
    # First, shift lnP's up so max lnP is zero, ensuring at least one nonzero P
    # Then shift lnP a little further to maximize number of non-underflowing values
    # without causing overflow
    # Key is only to add or subtract from lnP - that way any introduced factors cancel out
    # during normalize by sum(P)
    lnP = np.exp(lnP - np.max(lnP) + 1000*np.log(2) - np.log(lnP.size))
    lnP  /= np.sum(lnP)                                      # Normalize P's
    return lnP

def w_sample_var(val, wts, ws):
    w_var = w_variance(val, wts)
    return np.sqrt(ws*w_var)

def tf_driver(tf, xi, P):
    #print(tf)
    Pt = P / np.exp(tf)
    Pt = normalize(Pt)
    ws = np.sum(Pt**2)
    Q = w_sample_var(xi, Pt, ws)
    #print(Q)
    return -Q

def find_best_tf(xi, P, u0):
    opt = fmin(tf_driver, np.log(u0), args=(xi,P), full_output=True)
    return (np.exp(opt[0][0]), -opt[1])

def credible_interval(X, P):
    sort_ascending = np.argsort(X)
    X_s = X[sort_ascending]
    P_s = P[sort_ascending]
    s = np.cumsum(P_s)
    c025 = np.where(s < 0.025)[0][-1]
    c0975 = np.where(s > 0.975)[0][0]
    
    X_low = X_s[c025]
    X_high = X_s[c0975]
    print(f"{X_low} -- {X_high}")
    
    return X_low, X_high
    
def w_mean(var, wts):
    """Calculates the weighted mean"""
    return np.average(var, weights=wts)


def w_variance(var, wts):
    """Calculates the weighted variance"""
    return np.average((var - w_mean(var, wts))**2, weights=wts)


def w_skew(var, wts):
    """Calculates the weighted skewness"""
    return (np.average((var - w_mean(var, wts))**3, weights=wts) /
            w_variance(var, wts)**(1.5))

def w_kurtosis(var, wts):
    """Calculates the weighted kurtosis"""
    return (np.average((var - w_mean(var, wts))**4, weights=wts) /
            w_variance(var, wts)**(2))

def covariance(X, Y, weights):
    avgx = np.average(X, weights=weights)
    avgy = np.average(Y, weights=weights)
    covar = np.average((X - avgx)*(Y-avgy), weights=weights)
    
    return covar

import statsmodels.sandbox.distributions.extras as extras
import scipy.interpolate as interpolate

def generate_normal_four_moments(mu, sigma, skew, kurt, minX, maxX, size=10000):
   f = extras.pdf_mvsk([mu, sigma, skew, kurt])
   x = np.linspace(minX, maxX, num=500)
   y = [f(i) for i in x]
   yy = np.cumsum(y) / np.sum(y)
   inv_cdf = interpolate.interp1d(yy, x, fill_value="extrapolate")
   rr = np.random.rand(size)

   return inv_cdf(rr)
    
def marginalize_1D(P, axis_overrides, bin_count, SECONDARY_PARAMS, param, X):
    minX, maxX = axis_overrides[param]
    bin_ct = bin_count

    bins = np.arange(bin_ct+1)
    bins   = minX + (maxX-minX)*(bins)/bin_ct    # Get params
    
    marP, bins = np.histogram(X, weights=P, bins=bins, density=True)
    
    if SECONDARY_PARAMS[param] or "mu" in param:
        # Correct for nonuniform sampling
        marP_h, bins = np.histogram(X, bins=bins)
        marP_corr = np.zeros_like(marP)
        for k in range(len(marP)):
            if marP_h[k] != 0:
                marP_corr[k] = marP[k] / marP_h[k]

        area = np.sum(np.diff(bins)*marP_corr)
        marP_corr /= area
        
    else:
        marP_corr = marP
        
    return (marP_corr, bins)

def marginalize_2D(P, axis_overrides, bin_count, SECONDARY_PARAMS, param_names, X, Y):
    paramx, paramy = param_names
    minX, maxX = axis_overrides[paramx]
    minY, maxY = axis_overrides[paramy]
    
    bin_ctx = bin_count
    bin_cty = bin_count
    
    bins_x = np.arange(bin_ctx+1)
    bins_y = np.arange(bin_cty+1)

    bins_x   = minX + (maxX-minX)*(bins_x)/bin_ctx    # Get params
    bins_y   = minY + (maxY-minY)*(bins_y)/bin_cty

    im = np.histogram2d(X, Y, bins=[bins_x,bins_y], weights=P, density=True)
    im_h = np.histogram2d(X, Y, bins=[bins_x,bins_y], density=True)
    
    # marP_corr /= np.sum(marP_corr)
    marP_corr = im[0] * 1
    Y_corr, X_corr = np.meshgrid(bins_x, bins_y)
    
    return (marP_corr, X_corr, Y_corr)