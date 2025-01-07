#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 22:21:41 2023

@author: qianruixin
"""

import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt

def triu(a):
    # Extract the upper triangular elements of the matrix
    triulist = []
    for a1 in range(len(a)):
        for a2 in np.arange(a1+1, len(a)):
            triulist.append(a[a1, a2])
    return triulist

def calc_glm(spike_rate_A, spike_rate_B, population_rate):
    # Z-score normalization
    spike_rate_B_norm = (spike_rate_B - np.mean(spike_rate_B)) / np.std(spike_rate_B)
    population_rate_norm = (population_rate - np.mean(population_rate)) / np.std(population_rate)

    # Design matrix
    X = sm.add_constant(np.column_stack((spike_rate_B_norm, population_rate_norm)))

    # GLM Fit with a Poisson family and a log link function
    glm_poisson = sm.GLM(spike_rate_A, X, family=sm.families.Poisson(link=sm.families.links.log()))
    glm_result = glm_poisson.fit()

    # Get the coefficients for the coupling between the spike rates of cells A and B (beta)
    # and the coupling of cell A to the population spike rate (beta_pop)
    beta, beta_pop = glm_result.params[1], glm_result.params[2]

    return beta, beta_pop

with open('sessionsGC.txt') as f:
    content = f.readlines()
session_list = []
for s in range(len(content)):
    session_list.append(content[s].split(' ')[0])

prefix = '../../Dataset/GC1/results/'
for s in session_list:
    # if s!='8a50a33f7fd91df4':
    #     continue
    if s in ['8a50a33f7fd91df4','1f20835f09e28706']:
        continue
    Out = np.load(prefix + s + '_firing_MEC.npy', allow_pickle=True).item()
    df_f_all = Out['df_f']
    
    num_border = Out['num_border']
    num_grid = Out['num_grid']
    num_hd = Out['num_hd']
    num_ov = Out['num_ov'] 
    num_other = Out['num_other'] 
    
    types = ['grid', 'border', 'hd', 'ov', 'other']
    
    border_idx = np.arange(num_border)
    grid_idx = np.arange(num_border, num_border + num_grid)
    hd_idx = np.arange(num_border + num_grid, num_border + num_grid + num_hd)
    ov_idx = np.arange(num_border + num_grid + num_hd, num_border + num_grid + num_hd + num_ov)
    other_idx = np.arange(num_border + num_grid + num_hd + num_ov, num_border + num_grid + num_hd + num_ov + num_other)
    cellidx = {}
    cellidx['border'] = border_idx.tolist()
    cellidx['grid'] = grid_idx.tolist()
    cellidx['hd'] = hd_idx.tolist()
    cellidx['ov'] = ov_idx.tolist()
    cellidx['other'] = other_idx.tolist()
    Rs = np.corrcoef(df_f_all.T)
    
    Lag = 10  # s
    DeltaT = 0.025  # s
    HalfLagPoints = int(Lag / DeltaT / 2)
    num_neurons = df_f_all.shape[1]
    LagRs = np.corrcoef(df_f_all.T[:, HalfLagPoints:], df_f_all.T[:, :(-HalfLagPoints)])[:num_neurons, num_neurons:]
    
    RsGLM = np.ones((num_neurons, num_neurons))
    RsPoP = np.ones((num_neurons, num_neurons))
    population_rate = np.sum(df_f_all.T, axis=0)
    for n1 in range(num_neurons):
        for n2 in np.arange(n1+1, num_neurons):
            RsGLM[n1, n2], RsPoP[n1, n2] = calc_glm(df_f_all.T[n1, :], df_f_all.T[n2, :], population_rate)
            RsGLM[n2, n1] = RsGLM[n1, n2]
            RsPoP[n2, n1] = RsPoP[n1, n2] 
    for i in range(len(Rs)):
        Rs[i, i] = np.nan
        LagRs[i, i] = np.nan
    plt.figure()
    plt.subplot(131)
    plt.imshow(Rs)
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(RsGLM)
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(RsPoP)
    plt.colorbar()
    
    plt.figure()
    triu_Rs=triu(Rs)
    triu_RsGLM=triu(RsGLM)
    plt.scatter(triu_Rs,triu_RsGLM)
    
    np.savez(prefix + s + '_RsGLM.npz', RsGLM=RsGLM, RsPoP=RsPoP, LagRs=LagRs, Rs=Rs)