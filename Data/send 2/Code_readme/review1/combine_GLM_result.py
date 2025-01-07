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

def triu(a):
    # Extract the upper triangular elements of the matrix
    triulist = []
    for a1 in range(len(a)):
        for a2 in np.arange(a1+1, len(a)):
            triulist.append(a[a1, a2])
    return triulist

#GC1
path_GC1 = '../../Dataset/GC1/DetectCP_GLM/'
path_RsGLM = '../../Dataset/GC1/results/'
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


au1 = np.load("../../Dataset/GC1AUC.npz", allow_pickle=True)
GC1_aucs2 = au1['aus']


plt.figure(figsize=(6,3))
plt.subplot(121)
RsGLM = np.load(path_RsGLM + s + '_RsGLM.npz')['RsGLM']
Rs = np.load(path_RsGLM + s + '_RsGLM.npz')['Rs'] #
triu_Rs=triu(Rs)
triu_RsGLM=triu(RsGLM)
plt.scatter(triu_Rs,triu_RsGLM,c='k',s=4,alpha=0.5)
plt.xlabel('Correlation')
plt.ylabel('Correlation (GLM)')
plt.subplot(122)
plt.scatter(GC1_aucs2,GC1_aucs,s=20)
plt.plot([0.5, 0.8], [0.5, 0.8], linestyle='--', color='k')
plt.xlabel('AUC')
plt.ylabel('AUC (GLM)')
ticks = [ 0.5, 0.6, 0.7, 0.8]
plt.xticks(ticks)
plt.yticks(ticks)

plt.tight_layout()
plt.savefig('./compare_GLM',dpi=400)

