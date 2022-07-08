#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:10:38 2022

@author: cfai2304

IO functions for importing TRPL observations and exporting result likelihood data
"""
import sys
import csv
import os
import numpy as np

def get_data(exp_files, ic_flags, sim_flags, logger=None, scale_f=1e-23):
    """ Import observation .csv files - see Examples/*_Observations.csv """
    # 1e-23 [cm^-2 s^-1] to [nm^-2 ns^-1]
    all_e_data = []
    
    bval_cutoff = sys.float_info.min
    if logger is not None: logger.info("cutoff: {}".format(bval_cutoff))

    EARLY_CUT = ic_flags['time_cutoff']
    SELECT = ic_flags['select_obs_sets']
    NOISE_LEVEL = ic_flags['noise_level']

    LOG_PL = sim_flags['log_pl']
    NORMALIZE = sim_flags["self_normalize"]
    
    for exp_file in exp_files:
        t = []
        PL = []
        uncertainty = []
        with open(exp_file, newline='') as file:
            eof = False
            next_t = []
            next_PL = []
            next_uncertainty = []
            ifstream = csv.reader(file)
            count = 0
            for row in ifstream:
                if row[0] == "END":
                    eof = True
                    finished = True
                else:
                    finished = (float(row[0]) == 0 and len(next_t))
    
                if eof or finished:
                    # t=0 means we finished reading the current PL curve - preprocess and package it
                    next_t = np.array(next_t)
                    if NOISE_LEVEL is not None:
                        next_PL = (np.array(next_PL) + NOISE_LEVEL*np.random.normal(0, 1, len(next_PL))) * scale_f
    
                    else:
                        next_PL = np.array(next_PL) * scale_f
    
                    next_uncertainty = np.array(next_uncertainty) * scale_f
    
                    if NORMALIZE:
                        next_PL /= max(next_PL)
    
                    if logger is not None:
                        logger.info("PL curve #{} finished reading".format(len(t)+1))
                        logger.info("Number of points: {}".format(len(next_t)))
                        logger.info("Times: {}".format(next_t))
                        logger.info("PL values: {}".format(next_PL))
                    if LOG_PL:
                        if logger is not None:
                            logger.info("Num exp points affected by cutoff: {}".format(np.sum(next_PL < bval_cutoff)))
    
                        # Deal with noisy negative values before taking log
                        next_PL = np.abs(next_PL)
                        next_PL[next_PL < bval_cutoff] = bval_cutoff
    
                        next_uncertainty /= next_PL
                        next_uncertainty /= 2.3 # Since we use log10 instead of ln
                        next_PL = np.log10(next_PL)
    
                    t.append(next_t)
                    PL.append(next_PL)
                    uncertainty.append(next_uncertainty)
    
                    next_t = []
                    next_PL = []
                    next_uncertainty = []
    
                    count = 0
    
                if not eof:
                    if (EARLY_CUT is not None and float(row[0]) > EARLY_CUT):
                        pass
                    else: 
                        next_t.append(float(row[0]))
                        next_PL.append(float(row[1]))
                        next_uncertainty.append(float(row[2]))
                
                count += 1
    
        if SELECT is not None:
            all_e_data.append((np.array(t)[SELECT], np.array(PL)[SELECT], np.array(uncertainty)[SELECT]))
        else:
            all_e_data.append((t, PL, uncertainty))
            
    return all_e_data

def get_initpoints(init_file, ic_flags, scale_f=1e-21, logger=None):
    """ Import initial excitation .csv files - see Examples/*_Excitations.csv """
    SELECT = ic_flags['select_obs_sets']

    with open(init_file, newline='') as file:
        ifstream = csv.reader(file)
        initpoints = []
        for row in ifstream:
            if len(row) == 0: continue
            try:
                initpoints.append(np.array(row, dtype=float))
            except ValueError:
                if logger is not None:
                    logger.warning("Failed to read IE")
                continue
        
    if SELECT is not None:
        initpoints = np.array(initpoints)[SELECT]
    return np.array(initpoints, dtype=float) * scale_f

def export(out_filename, P, X, logger=None):
    """ Export list of likelihoods (*_BAYRAN_P.npy) and sample parameter points (*_BAYRAN_X.npy) """
    try:
        if logger is not None: logger.info("Creating dir {}".format(out_filename))
        os.mkdir(out_filename)
    except FileExistsError:
        if logger is not None:
            logger.warning("{} dir already exists".format(out_filename))

    try:
        if logger is not None: 
            logger.info("Writing to {}:".format(out_filename))
        base = os.path.basename(out_filename)
        np.save(os.path.join(out_filename, "{}_BAYRAN_P.npy".format(base)), P)
        np.save(os.path.join(out_filename, "{}_BAYRAN_X.npy".format(base)), X)
        
    except Exception as e:
        if logger is not None: logger.error("Error export {}".format(e))

    return

# def save_raw_pl(out_filename, ic_num, blk, plI):
#     """ DEPRECATED - save direct output of TRPL simulation """
#     try:
#         np.save(os.path.join(out_filename, "plI{}_grp{}.npy".format(ic_num, blk), plI))
#         print("Saved plI of size ", plI.shape)
#     except Exception as e:
#         print("Warning: save failed\n", e)
        
# def load_raw_pl(out_filename, ic_num, blk):
#     """ DEPRECATED - load direct output of TRPL simulation """
#     try:
#         plI = np.load(os.path.join(out_filename, "plI{}_grp{}.npy".format(ic_num, blk)))
#         print("Loaded plI of size ", plI.shape)
#     except Exception as e:
#         print("Error: load failed\n", e)
#         sys.exit()
#     return plI