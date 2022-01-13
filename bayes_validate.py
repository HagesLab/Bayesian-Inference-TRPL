#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:26:14 2022

@author: cfai2304
"""
from numba import cuda
def validate_IC(ics, L):
    for ic in ics:
        assert len(ic) == L, "Error: IC length:{}, declared L:{}".format(len(ic), L)
    return

def validate_ic_flags(ic_flags):
    if ic_flags["time_cutoff"] is not None:
        assert isinstance(ic_flags["time_cutoff"], (float, int)), "invalid time cutoff"
        assert ic_flags["time_cutoff"] > 0, "invalid time cutoff"

    if ic_flags["select_obs_sets"] is not None:
        assert isinstance(ic_flags["select_obs_sets"], list), "invalid observation set selection"

    if ic_flags["noise_level"] is not None:
        assert isinstance(ic_flags["noise_level"], (float, int)), "invalid noise level"
    return

def validate_gpu_info(gpu_info):
    assert isinstance(gpu_info["num_gpus"], int), "invalid num_gpus"
    assert gpu_info["num_gpus"] > 0, "invalid num_gpus"
    assert gpu_info["num_gpus"] <= 8, "too many gpus"

    assert isinstance(gpu_info["sims_per_gpu"], int), "invalid sims per gpu"
    assert gpu_info["sims_per_gpu"] > 0, "invalid sims per gpu"

    return

def validate_params(num_params, unit_conversions, do_log, minX, maxX):
    assert (len(unit_conversions) == num_params), "Unit conversion array is missing entries"
    assert (len(do_log) == num_params), "do_log mask is missing values"
    assert (len(minX) == num_params), "Missing min param values"
    assert (len(maxX) == num_params), "Missing max param values"
    assert all(minX <= maxX), "Min params larger than max params"
    return

def connect_to_gpu(gpu_info):

    print("Detecting GPU...")
    gpu_info["has_GPU"] = cuda.detect()
    if gpu_info["has_GPU"]:
        device = cuda.get_current_device()

        gpu_info["threads_per_block"] = (2 ** 7,)
        gpu_info["max_sims_per_block"] = 3           # Maximum of 6 due to shared memory limit
            

    return