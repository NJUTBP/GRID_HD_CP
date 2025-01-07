#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:35:29 2024

@author: qianruixin
"""

import pickle
import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt

import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

#HD
path_HD = '../../Dataset/HD/ProcessedData/DetectCP_max/'
IDs = ['Mouse12-120806', 'Mouse12-120807', 'Mouse12-120808', 'Mouse12-120809', 'Mouse12-120810',
      'Mouse20-130517', 'Mouse28-140313', 'Mouse25-140130']
Clus = ['1_8', '1_8', '1_8', '1_8', '1_8', '1_8', '8_11', '5_8']
States = ['Wake', 'REM', 'SWS']

auc=[]
labels=[]
HD_aucs=[]
for m in IDs:
    for s in States:
        with open(path_HD+m+'_'+s+'.pickle', 'rb') as file:
            data = pickle.load(file)
        HD_aucs.append(data['auc'])
        auc.append(data['auc'])
        labels.append('HD')
#GC1
path_GC1 = '../../Dataset/GC1/DetectCP_max/'
with open('sessionsGC.txt') as f:
    content = f.readlines()
session_list = []
for s in range(len(content)):
    session_list.append(content[s].split(' ')[0]) 

GC1_aucs=[]
for s in session_list:
    with open(path_GC1+s+'.pickle', 'rb') as file:
        data = pickle.load(file)
    GC1_aucs.append(data['auc'])
    auc.append(data['auc'])
    labels.append('GC1')

#GC2
path_GC2 = '../../Dataset/GC2/DetectCP_max/'
mice = [ 'Mumbai', 'Kerala', 'Goa', 'Punjab'] # cue rich
sessions = [['1201_1'],
            ['1207_1'], # Kerala
            ['1210_1'],
            ['1217_1']
            ]
GC2_aucs=[]
for m, session in zip(mice, sessions):
    for s in session:
        with open(path_GC2+m+'_'+s+'.pickle', 'rb') as file:
            data = pickle.load(file)
        GC2_aucs.append(data['auc'])
        auc.append(data['auc'])
        labels.append('GC2')

data = {'xlist': auc,
        'labels': labels}
df = pd.DataFrame(data)

plt.figure(figsize=(12,3))
ax1=plt.subplot(141)
cmap = ListedColormap(sns.color_palette('tab10'))
color_dict = {'HD': cmap(2), 'GC1': cmap(4), 'GC2': cmap(6)} # purple, pink, blue
sns.swarmplot(x='labels', y='xlist', data=df, palette=color_dict,ax=ax1)
ax1.set_xticks(ticks=[0, 1, 2], labels=[r'$\it{HD}$',r'$\it{GC-1}$',r'$\it{GC-2}$'])
ax1.set_xlabel('')
ax1.set_ylabel('AUC',labelpad=0.1)


au1 = np.load("../../Dataset/HDAUC.npz", allow_pickle=True)
HD_aucs2 = au1['aus']
au1 = np.load("../../Dataset/GC1AUC.npz", allow_pickle=True)
GC1_aucs2 = au1['aus']
au1 = np.load("../../Dataset/GC2AUC.npz", allow_pickle=True)
au1 = au1['aus']
GC2_aucs2=[]
# print(np.mean(au1))
# print(np.std(au1))
sele_idx = [1, 2, 4, 6]
for i in range(len(au1)):
    if i in sele_idx:
        # labels.append('GC2')
        GC2_aucs2.append(au1[i])

plt.subplot(142)
plt.scatter(HD_aucs,HD_aucs2)
plt.xlabel('AUC (max)')
plt.ylabel('AUC (aggregated)')

plt.subplot(143)
plt.scatter(GC1_aucs,GC1_aucs2)
plt.xlabel('AUC (max)')
plt.ylabel('AUC (aggregated)')

plt.subplot(144)
plt.scatter(GC2_aucs,GC2_aucs2)
plt.xlabel('AUC (max)')
plt.ylabel('AUC (aggregated)')

plt.tight_layout()
plt.savefig('./compare_max_aggre',dpi=400)

