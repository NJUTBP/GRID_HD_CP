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


    
# prefix = '../../Dataset/GC1/results/'
pathout = '../../Dataset/HD/ProcessedData/DetectCP_selecore/'

fig=plt.figure(figsize=(8.5,11))


# IDs = ['Mouse12-120806', 'Mouse12-120807', 'Mouse12-120808', 'Mouse12-120809', 'Mouse12-120810',
#       'Mouse20-130517', 'Mouse28-140313', 'Mouse25-140130']
# Clus = ['1_8', '1_8', '1_8', '1_8', '1_8', '1_8', '8_11', '5_8']
# States = ['Wake', 'REM', 'SWS']


# IDs = ['Mouse12-120806', 'Mouse12-120807', 'Mouse12-120808', 'Mouse12-120809',]
IDs = ['Mouse12-120810','Mouse20-130517', 'Mouse28-140313', 'Mouse25-140130',]
States = ['Wake', 'REM', 'SWS']
for no,m in enumerate(IDs):
    for no2,s in enumerate(States):
        aucs=[]
        core_nums=[]
        ratios=[0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        
        for ratio in ratios:
            with open(pathout+m+'_'+States[no2]+'_'+str(ratio)+'.pickle', 'rb') as file:
                data = pickle.load(file)
            core_scores=list(data['xlist_w_dict'].values())
            
            auc=data['auc']
            
            aucs.append(auc)
            core_nums.append(sum(1 for x in core_scores if x > 0.6))
        
        # 计算皮尔森相关系数和 p 值
        r1, p1 = pearsonr(ratios, aucs)
        r2, p2 = pearsonr(ratios, core_nums)
        # plt.figure(figsize=(6,3))
        
        ax1 = fig.add_axes([0.06 + 0.33 * (no2 % 3), 0.80 - 0.17 * no, 0.11, 0.08])
        ax1.scatter(ratios, aucs,s=9)
        ax1.set_xlabel('The ratio of selected HD cells',fontsize=5)
        ax1.set_ylabel('AUC')
        ax1.set_title(f'Pearson r: {r1:.2f}, p: {p1:.2g}')
        
        ax_pos = ax1.get_position()
        fig.text(ax_pos.x0-0.02, ax_pos.y1+0.025,m+'_'+s,c='purple')
        
        ax2 = fig.add_axes([0.21 + 0.33 * (no2 % 3), 0.80 - 0.17 * no, 0.11, 0.08])
        ax2.scatter(ratios, core_nums,s=9)
        ax2.set_xlabel('The ratio of selected HD cells',fontsize=5)
        ax2.set_ylabel('The number of core neurons',fontsize=5,labelpad=0.01)
        ax2.set_title(f'Pearson r: {r2:.2f}, p: {p2:.2g}')
        
        
plt.savefig('./combine_sele_core_HD2.png', dpi=400)
        
        
        
    
