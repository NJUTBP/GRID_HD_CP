#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:22:52 2023

@author: qianruixin
"""

import cpnet
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   
from pandas import  DataFrame
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

import sys
sys.path.append('../') # 将上级目录加入 sys.path
sys.path.append('../../') # 将上级目录加入 sys.path
from DetectCP import GenerateGraph, CalCoreness, Metrics, DrawNetwork
from DrawStandard import *
width = 100 / 25.4 

from sklearn.metrics import confusion_matrix
import pickle
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

with open('sessionsGC.txt') as f:
    content = f.readlines()
session_list = []
for s in range(len(content)):
    session_list.append(content[s].split(' ')[0]) # During the sessions, more than one grid cells were recorded.
    
prefix = '../../Dataset/GC1/results/'
pathout = '../../Dataset/GC1/DetectCP_GLM/'
AUCs=[]
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
    Rs = np.load(prefix + s + '_RsGLM.npz')['RsGLM'] # The diagonal is nan.
    LagRs = np.zeros((len(Rs),len(Rs)))
    
    cellidxother = cellidx['border'] + cellidx['hd'] + cellidx['ov'] + cellidx['other']
    cellidxall = cellidx['border'] + cellidx['grid'] + cellidx['hd'] + cellidx['ov'] + cellidx['other']

    # plt.figure(figsize = (width*4, width))
    # plt.subplot(1, 4, 1)
    # lagr
    # RsTemp=Rs-LagRs
    G = GenerateGraph(Rs, LagRs, CutPos=0.01, CutNeg=-0.01)
    xlist_w = CalCoreness(G)
    xlist_w_dict={}
    xstd = {}
    c = {}
    labels = []
    labelsnodes = {}
    colors_mark=[]  
    num_neurons = len(Rs)
    node_edgecolors = {}
    for n in range(num_neurons):  
        xlist_w_dict[n] = xlist_w[n]
        c[n] = 0
        labelsnodes[n] = cellids[n].split('_')[1]
        if n in grid_idx:
            xstd[n] = 1
            labels.append('g')
            colors_mark.append('#FFC0CB')
            node_edgecolors[n] = 2
        else:
            xstd[n] = 0
            labels.append('o')
            colors_mark.append('#87CEEB')  
            node_edgecolors[n] = 0
    data={'Coreness Score' : xlist_w,
    'Labels' : labels}
    df = DataFrame(data)
            
    import matplotlib.gridspec as gridspec

    # 创建一个1x5的网格布局
    gs = gridspec.GridSpec(1, 5)
    
    plt.figure(figsize = (width*4.5, width*0.95))
    # plt.figure(figsize = (width*5, width*1))
    
    ax = plt.subplot(gs[0])
    # ax, pos = DrawNetwork(G, c, xlist_w_dict, xstd, ax, _size=7, labels=labelsnodes) 
    ax, pos = DrawNetwork(G, c, xlist_w_dict, node_edgecolors, ax, draw_nodes_kwd={'linewidths':3.0}) 
    
    ax = plt.subplot(gs[1])
    sns.kdeplot(data=df, x='Coreness Score' , hue='Labels', common_norm=False, fill=True, cut=0, palette=['royalblue','darkorange'])
    plt.xlabel('Core score')
    plt.ylabel('Density')
    plt.gca().legend_.remove()
    
    # 注意这里使用的是gs[2:4]，这使得第四个子图跨越了两列并且前移
    ax = plt.subplot(gs[2:4])   
    
    threshold = 0.6
    
    # 根据阈值将预测得分转换为二进制分类预测
    y_true = [value for value in xstd.values()]
    y_scores = np.array(xlist_w)
    y_pred = np.where(y_scores >= threshold, 1, 0)
    
    ratios = compute_ratios(y_true, y_pred)
    x = np.arange(len(ratios))
    colors = ['pink', 'grey', 'pink', 'grey']
    bars = ax.bar(x, ratios, color=colors)
    ax.yaxis.grid(True, linestyle='--')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel('% of cells in class')
    yticks = [0, 0.25, 0.5, 0.75, 1]
    labels = ['0', '25', '50', '75', '100']
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    ax.set_title('{:3f}'.format(threshold))
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{:.0f}%'.format((height*100)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax = plt.subplot(gs[4])   
    # Calculate the points on the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, c='darkorange', label='ROC curve (AUC = %0.3f)' % auc)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing', c='royalblue',)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{:4f}'.format(auc))
    # plt.legend()
    
    plt.tight_layout()
    plt.savefig(pathout+s+'.png', dpi=300, bbox_inches='tight')
    
    output={}
    output['G']=G
    output['c']=c
    output['xlist_w_dict']=xlist_w_dict
    output['node_edgecolors']=node_edgecolors
    output['df']=df
    
    bars = ax.bar(x, ratios, color=colors)
    output['ratios']=ratios
    output['x']=x
    output['colors']=colors
    
    output['fpr']=fpr     
    output['tpr']=tpr     
    output['auc']=auc     
    with open(pathout+s+'.pickle', 'wb') as file: #w -> write; b -> binary
        pickle.dump(output, file)
# np.savez("/Volumes/BackupPlus/machine_identify_GC/Dataset/GC1AUC.npz", aus=AUCs)