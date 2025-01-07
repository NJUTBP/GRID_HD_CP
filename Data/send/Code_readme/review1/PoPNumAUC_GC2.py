#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:43:43 2023

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
sys.path.append('../') # 将上级目录加入 sys.path
from DetectCP_max import GenerateGraph, CalCoreness, Metrics, DrawNetwork
from DrawStandard import *
width = 100 / 25.4 
import pickle

from sklearn.metrics import confusion_matrix
def compute_ratios(y_true, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 计算比率
    ratio_tp = tp / (tp + fp)  # 判断为正确的类别中真正为正确的比率
    ratio_fp = fp / (tp + fp)  # 判断为正确的类别中真正为错误的比率
    ratio_fn = fn / (fn + tn)  # 判断为错误的类别中真正为正确的比率
    ratio_tn = tn / (fn + tn)  # 判断为错误的类别中真正为错误的比率
    
    ratios = [ratio_tp, ratio_fp, ratio_fn, ratio_tn]
    return ratios


def triu(a):
    # Extract the upper triangular elements of the matrix
    triulist = []
    for a1 in range(len(a)):
        for a2 in np.arange(a1+1, len(a)):
            triulist.append(a[a1, a2])
    return triulist


base = '../../Dataset/GC2/'
data_folder = base + 'aggregate_data/'

pathout = '../../Dataset/GC2/DetectCP_max/'
''' 9 sessions'''
# mice = ['Mumbai', 'Kerala', 'Goa', 'Punjab', 'Salvador'] # cue rich
# mouse_IDs = ['9a', ' 9b', '9c', '9d', '10a']
# sessions = [['1130_1', '1201_1'], # Mumbai
#             ['1207_1'], # Kerala
#             ['1211_1', '1210_1', '1209_1'], # Goa
#             ['1217_1', '1214_1'], # Punjab
#             ['1202_1'] # Salvador
#             ]

mice = [ 'Mumbai', 'Kerala', 'Goa', 'Punjab'] # cue rich
sessions = [['1201_1'],
            ['1207_1'], # Kerala
            ['1210_1'],
            ['1217_1']
            ]

pop_nums=[]
aucs_all=np.load("../../Dataset/GC2AUC.npz")['aus']
aucs=[]
sele_idx = [1, 2, 4, 6]
for i in range(len(aucs_all)):
    if i in sele_idx:
        aucs.append(aucs_all[i])
for m, session in zip(mice, sessions):
    for s in session:
        cellidx = np.load(data_folder + '/gain_manip/' + str(m) + '_' + str(s) + '_MEC_cellidx.npy', allow_pickle=True).item()
                        
        Rs = np.load(data_folder + '/gain_manip/' + str(m) + '_' + str(s) + '_trialall_Rs.npz')['Rs']
        LagRs = np.load(data_folder + '/gain_manip/' + str(m) + '_' + str(s) + '_trialall_LagRs10.npz')['LagRs']

        types = ['grid', 'border', 'inter', 'other_spatial', 'nonspatial']
        num_grid = len(cellidx['grid'])

        cellidxother = cellidx['border'] + cellidx['other_spatial'] + cellidx['nonspatial']
        cellidxall = cellidx['grid'] + cellidx['border'] + cellidx['other_spatial'] + cellidx['nonspatial']
        
        pop_nums.append(len(cellidxall))

plt.figure()
plt.scatter(pop_nums,aucs)

np.savez("./PoPNumAUC_GC2", aus=aucs, pop_nums=pop_nums)
       
       