#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 18:26:14 2022

@author: cfai2304
"""

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