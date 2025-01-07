#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:51:13 2024

@author: qianruixin
"""

import pickle
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')

import sys
sys.path.append('./') 
from DrawStandard import *

with open('sessionsGC.txt') as f:
    content = f.readlines()
session_list = []
for s in range(len(content)):
    session_list.append(content[s].split(' ')[0]) # During the sessions, more than one grid cells were recorded.
    
prefix = '../../Dataset/GC1/results/'
pathout = '../../Dataset/GC1/DetectCP_selecore/'

fig=plt.figure(figsize=(8.5,11))

for no,s in enumerate(session_list):
    aucs=[]
    core_nums=[]
    ratios=[0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    for ratio in ratios:
        with open(pathout+s+'_'+str(ratio)+'.pickle', 'rb') as file:
            data = pickle.load(file)
        core_scores=list(data['xlist_w_dict'].values())
        
        auc=data['auc']
        
        aucs.append(auc)
        core_nums.append(sum(1 for x in core_scores if x > 0.6))
    
    # 计算皮尔森相关系数和 p 值
    r1, p1 = pearsonr(ratios, aucs)
    r2, p2 = pearsonr(ratios, core_nums)
    # plt.figure(figsize=(6,3))
    
    ax1 = fig.add_axes([0.1 + 0.43 * (no % 2), 0.80 - 0.18  * int(no / 2), 0.15, 0.1])
    ax1.scatter(ratios, aucs,s=9)
    ax1.set_xlabel('The ratio of selected grid cells')
    ax1.set_ylabel('AUC')
    ax1.set_title(f'Pearson r: {r1:.2f}, p: {p1:.2g}')
    
    ax_pos = ax1.get_position()
    fig.text(ax_pos.x0-0.05, ax_pos.y1+0.025, s,c='purple')
    
    ax2 = fig.add_axes([0.30 + 0.43 * (no % 2), 0.80 - 0.18 * int(no / 2), 0.15, 0.1])
    ax2.scatter(ratios, core_nums,s=9)
    ax2.set_xlabel('The ratio of selected grid cells')
    ax2.set_ylabel('The number of core neurons')
    ax2.set_title(f'Pearson r: {r2:.2f}, p: {p2:.2g}')
    
    
plt.savefig('./combine_sele_core_GC1.png', dpi=400)
    
    
    

