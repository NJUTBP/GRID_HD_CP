#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:22:52 2023

@author: qianruixin
"""

import cpnet
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt   
from pandas import  DataFrame
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

import sys
sys.path.append('../../') # 将上级目录加入 sys.path
from DetectCP import GenerateGraph, CalCoreness, Metrics, DrawNetwork
from DrawStandard import *

def triu(a):
    # Extract the upper triangular elements of the matrix
    triulist = []
    for a1 in range(len(a)):
        for a2 in np.arange(a1+1, len(a)):
            triulist.append(a[a1, a2])
    return triulist

with open('sessionsGC.txt') as f:
    content = f.readlines()
session_list = []
for s in range(len(content)):
    session_list.append(content[s].split(' ')[0]) # During the sessions, more than one grid cells were recorded.
# session_list = ['e64acc09dc77a53c']
prefix = '../../Dataset/GC1/results/'
pop_nums=[]
for s in session_list:
    # if s!='8a50a33f7fd91df4':
    #     continue
    Out = np.load(prefix + s + '_firing_MEC.npy', allow_pickle=True).item()
    df_f_all = Out['df_f']
    
    num_border = Out['num_border']
    num_grid = Out['num_grid']
    num_hd = Out['num_hd']
    num_ov = Out['num_ov'] 
    num_other = Out['num_other'] 
    
    cellids = Out['cell_id'] 
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
    Rs = np.load(prefix + s + '_Rs.npz')['Rs'] # The diagonal is nan.
    LagRs = np.load(prefix + s + '_Rs.npz')['LagRs10']
    
    cellidxother = cellidx['border'] + cellidx['hd'] + cellidx['ov'] + cellidx['other']
    cellidxall = cellidx['border'] + cellidx['grid'] + cellidx['hd'] + cellidx['ov'] + cellidx['other']

    pop_num=len(cellidxall)
    pop_nums.append(pop_num)
# np.savez("../../../Dataset/GC1AUC.npz", aus=AUCs)
aucs=np.load("../../Dataset/GC1AUC.npz")['aus']

plt.figure()
plt.scatter(pop_nums,aucs)

np.savez("./PoPNumAUC_GC1", aus=aucs, pop_nums=pop_nums)

'''
loaded_data = np.load("../../../Dataset/GC1AUC.npz")
loaded_aucs = loaded_data['aus']
loaded_pop_nums = loaded_data['pop_nums']
'''



