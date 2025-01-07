# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 2023

@author: Tao WANG

Description: Draw Fig1 of RTO
"""

####加载库####
import numpy as np
np.random.seed(20231230)
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colors as mpcolors
from matplotlib.cm import Blues
from matplotlib import cm  # Import the cm module
import matplotlib.ticker as mtick
from matplotlib.patches import Circle
from statannotations.Annotator import Annotator
from scipy import stats
from scipy.stats import pearsonr

import random
import networkx as nx
import sys
sys.path.append('../') # 将上级目录加入 sys.path
from DrawStandard import *
import pickle
from DetectCP import GenerateGraph, CalCoreness, Metrics, DrawNetwork
import seaborn as sns

from sklearn.linear_model import RANSACRegressor

sys.path.append('.')
from draw_diamond import *

font = {
        'family' : 'arial',#'monospace',#'Times New Roman',#
        #'weight' : 'bold',
        'size'   : 6,
}
mathtext = {
        'fontset' : 'dejavusans',#'stix',
}
lines = {
        'linewidth' : 1.5,
}
xtick = {
        'direction' : 'out',
        'major.size' : 2,
        'major.width' : 1,
        'minor.size' : 1,
        'minor.width' : 0.5,
        'labelsize' : 6,
}
ytick = {
        'direction' : 'out',
        'major.size' : 2,
        'major.width' : 1,
        'minor.size' : 1,
        'minor.width' : 0.5,
        'labelsize' : 6,
}
axes = {
        'linewidth' : 1,
        'titlesize' : 6,
        #'titleweight': 'bold',
        'labelsize' : 6,
        #'labelweight' : 'bold',
}

matplotlib.rc('font',**font)
matplotlib.rc('mathtext',**mathtext)
matplotlib.rc('lines',**lines)
matplotlib.rc('xtick',**xtick)
matplotlib.rc('ytick',**ytick)
matplotlib.rc('axes',**axes)

def main():
    fig=plt.figure(figsize=(8.5,11))
    ax1=fig.add_axes([0.05,0.84,0.20,0.10])
    ax2=fig.add_axes([0.05,0.69,0.20,0.10])

    axa=fig.add_axes([0.30,0.84,0.10*11/8.5,0.10])
    axb=fig.add_axes([0.30,0.69,0.10*11/8.5,0.10])
    axc=fig.add_axes([0.48,0.84,0.12,0.1])
    axd=fig.add_axes([0.48,0.69,0.18,0.1])

    ax71=fig.add_axes([0.05,0.53,0.13,0.13*8.5/11])
    ax72=fig.add_axes([0.205,0.53,0.13,0.13*8.5/11])
    ax73=fig.add_axes([0.36,0.53,0.13,0.13*8.5/11])
    ax74=fig.add_axes([0.515,0.53,0.13,0.13*8.5/11])
    ax81=fig.add_axes([0.695,0.53,0.13,0.13*8.5/11])
    ax82=fig.add_axes([0.85,0.53,0.13,0.13*8.5/11])

    axX1=fig.add_axes([0.05,0.37,0.16,0.08])
    axX2=fig.add_axes([0.24,0.37,0.16,0.08])
    axY1=fig.add_axes([0.05,0.45,0.16,0.04])
    axY2=fig.add_axes([0.24,0.45,0.16,0.04])
    axZ1=fig.add_axes([0.46,0.37,0.12,0.12])
    # axZ2=fig.add_axes([0.77,0.37,0.04,0.12])
    axZ3=fig.add_axes([0.62,0.37,0.18,0.12])

    path1='../Data/CoreNeuron/HD/cellpair_spr/'
    m='Mouse12-120806'
    states=['Wake','REM','SWS']
    statenames=['RUN','REM','SWS']
    marker=['s','o','^','d']
    colors=['tab:blue','tab:orange','tab:green','tab:red']
    for no,s in enumerate(states):
        with open(path1+m+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        dists_s=output['dists_s']
        r0_s=output['r0_s']
        ax1.scatter(r0_s, dists_s, marker=marker[no],edgecolor=colors[no],facecolor='none', alpha=0.5,s=7,label=statenames[no])
    ax1.legend(facecolor='none',edgecolor='none',title=r'$\it{HD}$',labelspacing=0.5,handletextpad=0.1)
    ax1.set_xlabel('Correlation coefficient',labelpad=2)
    ax1.set_ylabel('$\\Delta \\theta$ (rad)',labelpad=0.1)
    # ax1.set_title(r'$\it{HD}$',pad=0.1)

    path2='../Data/CoreNeuron/GC1/cellpair_spr_core/'
    s='5b92b96313c3fc19'
    with open(path2+s+'.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    dists_s=output['dists_s']
    r0_s=output['r0_s']
    ax2.scatter(r0_s, dists_s, edgecolor=colors[0],facecolor='none', alpha=0.5,s=7)
    ax2.set_xlabel('Correlation coefficient',labelpad=2)
    ax2.set_ylabel('$|\\Delta \\bf{g}|$ (rad)',labelpad=0.1)
    # ax2.set_title(r'$\it{GC-1}$',pad=0.1)
    ax2.text(0.6,3.5,r'$\it{GC-1}$')

    path4='../Data/CoreNeuron/HD/predict_d/'    
    m='Mouse12-120806'
    s='Wake'
    with open(path4+m+'_'+s+'.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    r2=output['r2']
    d2=output['d2']
    r1=output['r1']
    r3=output['r3']
    d1=output['d1']
    d3=output['d3']

    axa.scatter(d1, d3, c='tab:blue', alpha=0.5,s=7)
    axa.plot([0,3.15], [0,3.15], c='grey', linewidth=2)
    axa.set_xlim([0, 3.15])
    axa.set_ylim([0, 3.15])
    axa.set_yticks(axa.get_xticks()[:-1])
    axa.set_xlabel('Actual $\\Delta \\theta$ (rad)',labelpad=0.1)
    axa.set_ylabel('Predicted $\\Delta \\theta$ (rad)',labelpad=0.1)  
    axa.set_title('one session from $HD$\n$r$ = {:.2f}'.format(pearsonr(d1, d3)[0]),pad=2)

    path5='../Data/CoreNeuron/GC1/predict_d/'    
    s='5b92b96313c3fc19'
    with open(path5+s+'.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    r2=output['r2']
    d2=output['d2']
    r1=output['r1']
    r1_all=output['r1_all']
    r3=output['r3']
    r3_all=output['r3_all']
    d1=output['d1']
    d1_all=output['d1_all']
    d3=output['d3']
    d3_all=output['d3_all']

    axb.scatter(d1, d3, c='tab:orange', alpha=0.5,s=7)
    axb.plot([0,4.5], [0,4.5], c='grey', linewidth=2)
    axb.set_xlim([0, 4.5])
    axb.set_ylim([0, 4.5])
    axb.set_yticks(axb.get_xticks()[:-1])
    axb.set_xlabel('Actual $|\\Delta \\bf{g}|$ (rad)',labelpad=0.1)
    axb.set_ylabel('Predicted $|\\Delta \\bf{g}|$ (rad)',labelpad=0.1)  
    axb.set_title('one session from $GC$-1\n$r$ = {:.2f}'.format(pearsonr(d1, d3)[0]),pad=2)

    rs_Wake=[]
    rs_SWS=[]
    rs_REM=[]
    path4='../Data/CoreNeuron/HD/predict_d/'
    Mice=['Mouse12-120806',
    'Mouse12-120807',
    'Mouse12-120808',
    'Mouse12-120809',
    'Mouse12-120810',
    'Mouse20-130517',
    'Mouse25-140130',
    'Mouse28-140313']
    States=['Wake','REM','SWS']
    for m in Mice:
        # m='Mouse12-120806'
        s='Wake'
        with open(path4+m+'_'+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        d1=output['d1']
        d3=output['d3']
        rs_Wake.append(pearsonr(d1, d3)[0])

        s='REM'
        with open(path4+m+'_'+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        d1=output['d1']
        d3=output['d3']
        rs_REM.append(pearsonr(d1, d3)[0])

        s='SWS'
        with open(path4+m+'_'+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        d1=output['d1']
        d3=output['d3']
        rs_SWS.append(pearsonr(d1, d3)[0])

    rs={'Wake':rs_Wake,'REM':rs_REM,'SWS':rs_SWS}

    cmap = ListedColormap(sns.color_palette('tab10'))
    color_dict = {'Wake': cmap(2), 'REM': cmap(4), 'SWS': cmap(6)} # purple, pink, blue
    # sns.swarmplot(x='labels', y='xlist', data=df, palette=color_dict)
    df = pd.DataFrame(rs)
    sns.swarmplot(data=df,palette=color_dict,ax=axc)
    axc.set_ylabel('$r$',fontsize=9)
    axc.set_xticklabels(['RUN\n' + r'$\it{HD}$','REM\n' + r'$\it{HD}$','SWS\n' + r'$\it{HD}$'])

    print(df.keys())
    axc.bar(0,np.mean(df['Wake']),yerr=np.std(df['Wake']),color=cmap(2),alpha=0.5,zorder=5)
    axc.bar(1,np.mean(df['REM']),yerr=np.std(df['REM']),color=cmap(4),alpha=0.5,zorder=5)
    axc.bar(2,np.mean(df['SWS']),yerr=np.std(df['SWS']),color=cmap(6),alpha=0.5,zorder=5)
    
    path5='../Data/GC1/predict_d/'  
    rs=[]  
    Subjects=['0de4b55d27c9f60f',
    '1f20835f09e28706',
    '5b92b96313c3fc19',
    '7e888f1d8eaab46b',
    '8a50a33f7fd91df4',
    '8f7ddffaf4a5f4c5',
    '59825ec5641c94b4',
    'c221438d58a0b796',
    ]
    # s='5b92b96313c3fc19'
    for s in Subjects:
        with open(path5+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        d1=output['d1']
        d3=output['d3']
        rs.append(pearsonr(d1, d3)[0])
    data_d={'OF\n' + r'$\it{GC-1}$':rs}

    cmap = ListedColormap(sns.color_palette('tab20'))
    color_dict = {'OF\n' + r'$\it{GC-1}$':cmap(4),'OF\n' + r'$\it{GC-3}$': cmap(5), 'WW\n' + r'$\it{GC-3}$': cmap(2), 'REM\n' + r'$\it{GC-3}$': cmap(8),'SWS\n' + r'$\it{GC-3}$':cmap(12)} 
    data = []
    for key, values in data_d.items():
        data.extend([(key, value) for value in values])

    print(data_d['OF\n$\\it{GC-1}$'])
    # 使用swarmplot绘制散点图
    color_palette = sns.color_palette('tab10', n_colors=len(data_d))
    sns.swarmplot(x=[item[0] for item in data], y=[item[1] for item in data],palette=color_dict, size=5,ax=axd)
    axd.set_xticklabels(data_d.keys())
    axd.set_ylabel('$r$',fontsize=9)
    # axd.legend(frameon=False)
    axd.bar(0,np.mean(data_d['OF\n' + r'$\it{GC-1}$']),yerr=np.std(data_d['OF\n' + r'$\it{GC-1}$']),color=cmap(4),alpha=0.5,zorder=5)
    
    axd.set_xlim(-0.5,4.5)

    axd.set_xticklabels(['OF\n' + r'$\it{GC}$-1'])


    path7='../Data/CoreNeuron/HD/spr_corrected/'
    m='Mouse12-120806'
    with open(path7+m+'Wake'+'_align.pickle', 'rb') as file: #w -> write; b -> binary
        output71=pickle.load(file) 
    pos71a=output71['pos1']
    pos71b=output71['pos2']
    dist71=output71['dist2']
    errs_pos2=output71['errs_pos2']
    errs_pos2_min=[min(errs_pos2)]
    errs_posr=output71['errs_posr']
    print(errs_pos2_min,np.mean(dist71))

    with open(path7+m+'REM'+'_align.pickle', 'rb') as file: #w -> write; b -> binary
        output72=pickle.load(file) 
    pos72a=output72['pos1']
    pos72b=output72['pos2']
    dist72=output72['dist2']
    with open(path7+m+'SWS'+'_align.pickle', 'rb') as file: #w -> write; b -> binary
        output73=pickle.load(file) 
    pos73a=output73['pos1']
    pos73b=output73['pos2']
    dist73=output73['dist2']
    
    posrs=output71['posrs']
    pos_random=posrs[0,:,:]
    dist7r=output71['distr']
    
    # # Draw grey circle
    # outer_circle1 = Circle((0, 0), radius=1, facecolor='none', edgecolor='lightgrey', linewidth=1.5, alpha=0.7)
    # outer_circle2 = Circle((0, 0), radius=1, facecolor='none', edgecolor='lightgrey', linewidth=1.5, alpha=0.7)
    # outer_circle3 = Circle((0, 0), radius=1, facecolor='none', edgecolor='lightgrey', linewidth=1.5, alpha=0.7)
    # outer_circle4 = Circle((0, 0), radius=1, facecolor='none', edgecolor='lightgrey', linewidth=1.5, alpha=0.7)
    # ax71.add_patch(outer_circle1)
    # ax72.add_patch(outer_circle2)
    # ax73.add_patch(outer_circle3)
    # ax74.add_patch(outer_circle4)
    
    size=12
    alpha=0.7
    # ax71.scatter(pos71a[:, 0], pos71a[:, 1], c='darkgray', s=size)    
    # ax71.scatter(pos71b[:, 0], pos71b[:, 1], c='tab:blue',s=size,alpha=alpha)  
    # ax72.scatter(pos72a[:, 0], pos72a[:, 1], c='darkgray', s=size)    
    # ax72.scatter(pos72b[:, 0], pos72b[:, 1], c='tab:orange',s=size,alpha=alpha)  
    # ax73.scatter(pos73a[:, 0], pos73a[:, 1], c='darkgray', s=size)    
    # ax73.scatter(pos73b[:, 0], pos73b[:, 1], c='tab:green',s=size,alpha=alpha)  
    # ax74.scatter(pos71a[:, 0], pos71a[:, 1], c='darkgray',s=size)    
    # ax74.scatter(pos_random[:, 0], pos_random[:, 1], c='tab:red',s=size,alpha=alpha)  
    
    # # Draw lines between pos1 and aligned_pos2
    # for n in range(len(pos71a)):
    #     ax71.plot([pos71a[n, 0], pos71b[n, 0]], [pos71a[n, 1], pos71b[n, 1]], c='grey', linewidth=0.6)
    # for n in range(len(pos72a)):
    #     ax72.plot([pos72a[n, 0], pos72b[n, 0]], [pos72a[n, 1], pos72b[n, 1]], c='grey', linewidth=0.6)
    # for n in range(len(pos73a)):
    #     ax73.plot([pos73a[n, 0], pos73b[n, 0]], [pos73a[n, 1], pos73b[n, 1]], c='grey', linewidth=0.6)
    # for n in range(len(pos71a)):
    #     ax74.plot([pos71a[n, 0], pos_random[n, 0]], [pos71a[n, 1], pos_random[n, 1]], c='grey', linewidth=0.6)
    
    # # ax71.set_aspect('equal', 'box')
    # ax71.axis('off')
    # ax72.axis('off')
    # ax73.axis('off')
    # ax74.axis('off')
    ax71.scatter(np.arctan2(pos71a[:, 0],pos71a[:, 1])+np.pi,np.arctan2(pos71b[:, 0],pos71b[:, 1])+np.pi,c='tab:blue',)
    ax72.scatter(np.arctan2(pos72a[:, 0],pos72a[:, 1])+np.pi,np.arctan2(pos72b[:, 0],pos72b[:, 1])+np.pi,c='tab:orange',)
    ax73.scatter(np.arctan2(pos73a[:, 0],pos73a[:, 1])+np.pi,np.arctan2(pos73b[:, 0],pos73b[:, 1])+np.pi,c='tab:green',)
    ax74.scatter(np.arctan2(pos71a[:, 0],pos71a[:, 1])+np.pi,np.arctan2(pos_random[:, 0],pos_random[:, 1])+np.pi,c='tab:red',)
    ax71.set_xlim(0,2*np.pi)
    ax71.set_xticks([0,np.pi,2*np.pi])
    ax71.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax71.set_ylim(0,2*np.pi)
    ax71.set_yticks([0,np.pi,2*np.pi])
    ax71.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax71.set_xlabel('Actual $\\theta$ (rad)',labelpad=0.1)
    ax71.set_ylabel('Estimated $\\theta$ (rad)',labelpad=0.1)
    ax72.set_xlim(0,2*np.pi)
    ax72.set_xticks([0,np.pi,2*np.pi])
    ax72.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax72.set_ylim(0,2*np.pi)
    ax72.set_yticks([0,np.pi,2*np.pi])
    ax72.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax72.set_xlabel('Actual $\\theta$ (rad)',labelpad=0.1)
    # ax72.set_ylabel('Estimated $\\theta$ (rad)',labelpad=0.1)
    ax73.set_xlim(0,2*np.pi)
    ax73.set_xticks([0,np.pi,2*np.pi])
    ax73.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax73.set_ylim(0,2*np.pi)
    ax73.set_yticks([0,np.pi,2*np.pi])
    ax73.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax73.set_xlabel('Actual $\\theta$ (rad)',labelpad=0.1)
    # ax73.set_ylabel('Estimated $\\theta$ (rad)',labelpad=0.1)
    ax74.set_xlim(0,2*np.pi)
    ax74.set_xticks([0,np.pi,2*np.pi])
    ax74.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax74.set_ylim(0,2*np.pi)
    ax74.set_yticks([0,np.pi,2*np.pi])
    ax74.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax74.set_xlabel('Actual $\\theta$ (rad)',labelpad=0.1)
    # ax74.set_ylabel('Estimated $\\theta$ (rad)',labelpad=0.1)

    ax71.set_title(r'$\it{HD}$ '+'RUN', fontsize=8,pad=0.1)
    ax72.set_title(r'$\it{HD}$ '+'REM', fontsize=8,pad=0.1)
    ax73.set_title(r'$\it{HD}$ '+'SWS', fontsize=8,pad=0.1)
    ax74.set_title(r'$\it{HD}$ '+'Random', fontsize=8,pad=0.1)


    path9='../Data/CoreNeuron/GC1/spr/'
    s='0de4b55d27c9f60f'
    with open(path9+s+'_align.pickle', 'rb') as file: #w -> write; b -> binary
        output81=pickle.load(file)
    pos81a=output81['pos1']
    pos81b=output81['pos2']
    posrs=output81['posrs']

    dist81=output81['dist2']
    dist8r=output81['distr']

    pos_random=posrs[0,:,:]

    pos81b += np.pi
    pos81a += np.pi
    pos81a = np.clip(pos81a, 0, 2*np.pi)
    pos81b = np.clip(pos81b, 0, 2*np.pi)
    # plot_phase_distribution(pos81a, 'darkgray', pos81b, 'tab:blue', ax81, s=12)
    # ax81.set_title(s, fontsize=8)
    ax81.scatter(pos81a[:,0],pos81b[:,0],marker='o',edgecolor='none',facecolor='tab:blue',alpha=0.5)
    ax81.scatter(pos81a[:,1],pos81b[:,1],marker='o',edgecolor='tab:blue',facecolor='none',alpha=0.5)
    
    
    ax81.set_title(r'$\it{GC}$-1 '+'OF', fontsize=8,pad=0.1)
    # Right subplot
    # plot_phase_distribution(pos81a, 'darkgray', posrs[0,:,:], 'tab:red', ax82, s=12)
    ax82.scatter(pos81a[:,0],posrs[0,:,0],marker='o',edgecolor='none',facecolor='tab:red',alpha=0.5)
    ax82.scatter(pos81a[:,1],posrs[0,:,1],marker='o',edgecolor='tab:red',facecolor='none',alpha=0.5)
    
    ax82.set_title(r'$\it{GC}$-1 '+'Random', fontsize=8,pad=0.1)
    ax81.set_xlim(0,2*np.pi)
    ax81.set_xticks([0,np.pi,2*np.pi])
    ax81.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax81.set_ylim(0,2*np.pi)
    ax81.set_yticks([0,np.pi,2*np.pi])
    ax81.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax81.set_xlabel('Actual $g_1$ or $g_2$ (rad)',labelpad=0.1)
    ax81.set_ylabel('Estimated $g_1$ or $g_2$ (rad)',labelpad=0.1)
    ax82.set_xlim(0,2*np.pi)
    ax82.set_xticks([0,np.pi,2*np.pi])
    ax82.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax82.set_ylim(0,2*np.pi)
    ax82.set_yticks([0,np.pi,2*np.pi])
    ax82.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax82.set_xlabel('Actual $g_1$ or $g_2$ (rad)',labelpad=0.1)
    # ax82.set_ylabel('Estimated $g_1$ or $g_2$ (rad)',labelpad=0.1)
    
    # 将范围划分为相等的间隔
    min_value = 0
    max_value = max(dist71.max(), dist7r.max())
    num_bins=500
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    for t in range(dist7r.shape[0]):   
        # print(t)
        x2, y2 = compute_cdf(dist7r[t, :], bin_edges)
        if y2[0] != 0:
            x2 = np.insert(x2, 0, 0)
            y2 = np.insert(y2, 0, 0)        
        
        axX1.plot(x2, y2, c='tab:red',alpha=0.2)
    # 计算 dist1 和 dist2 的 CDF

    x1, y1 = compute_cdf(dist71, bin_edges)      
    if y1[0] != 0:
        x1 = np.insert(x1, 0, 0)
        y1 = np.insert(y1, 0, 0)
    axX1.plot(x1, y1, label='dist1', c='tab:blue') 

    x1, y1 = compute_cdf(dist72, bin_edges)      
    if y1[0] != 0:
        x1 = np.insert(x1, 0, 0)
        y1 = np.insert(y1, 0, 0)
    axX1.plot(x1, y1, label='dist1', c='tab:orange') 

    x1, y1 = compute_cdf(dist73, bin_edges)      
    if y1[0] != 0:
        x1 = np.insert(x1, 0, 0)
        y1 = np.insert(y1, 0, 0)

    axX1.plot(x1, y1, label='dist1', c='tab:green') 
    
    sns.kdeplot(data=np.mean(dist7r,axis=1),ax=axY1,c='tab:red')
    axY1.vlines(np.mean(dist7r,axis=1),0,1,color='tab:red',alpha=0.2)
    axY1.vlines(np.mean(dist71),0,2, color='tab:blue')
    axY1.vlines(np.mean(dist72),0,2, color='tab:orange')
    axY1.vlines(np.mean(dist73),0,2, color='tab:green')

    axX1.set_xlim(-0.1,3.2)
    axY1.set_xlim(-0.1,3.2)
    axX1.set_xticks([0,1.0,2.0,3.0])
    axY1.set_xticks([0,1.0,2.0,3.0],[])
    axX1.set_ylim(0,1.15)
    axY1.set_ylim(0,3.5)
    axX1.set_yticks([0,0.5,1.0])
    axY1.set_yticks([0,1,2,3])
    axX1.set_xlabel('Estimation error (rad)',labelpad=0.1)
    axX1.set_ylabel('Cumulative fraction',labelpad=0.1)
    axY1.set_ylabel('PDF',labelpad=6)

    min_value = 0
    max_value = max(dist81.max(), dist8r.max())
    # 将范围划分为相等的间隔
    num_bins=500
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)

    for t in range(dist8r.shape[0]):   
        # print(t)
        x2, y2 = compute_cdf(dist8r[t, :], bin_edges)
        if y2[0] != 0:
            x2 = np.insert(x2, 0, 0)
            y2 = np.insert(y2, 0, 0)        
        
        axX2.plot(x2, y2, c='tab:red',alpha=0.2)
    # 计算 dist1 和 dist2 的 CDF
    x1, y1 = compute_cdf(dist81, bin_edges)      
    if y1[0] != 0:
        x1 = np.insert(x1, 0, 0)
        y1 = np.insert(y1, 0, 0)
    axX2.plot(x1, y1, label='dist1', c='tab:blue') 

    sns.kdeplot(data=np.mean(dist8r,axis=1),ax=axY2,color='tab:red')
    axY2.vlines(np.mean(dist8r,axis=1),0,1.7,color='tab:red',alpha=0.2)
    axY2.vlines(np.mean(dist81),0,3.4,color='tab:blue')

    axX2.set_xlim(-0.1,4.5)
    axY2.set_xlim(-0.1,4.5)
    axX2.set_xticks([0,1,2,3,4])
    axY2.set_xticks([0,1,2,3,4],[])
    axX2.set_ylim(0,1.15)
    axY2.set_ylim(0,5.5)
    axX2.set_yticks([0,0.5,1.0])
    axY2.set_yticks([0,2,4])
    axX2.set_xlabel('Estimation error (rad)',labelpad=0.1)
    # axX2.set_ylabel('Cumulative fraction',labelpad=0.1)
    axY2.set_ylabel(None)

    pathX='../Data/CoreNeuron/HD/spr_corrected/'
    r_list=['12-120806','12-120807','12-120808','12-120809','12-120810','20-130517','25-140130','28-140313']
    # s='0de4b55d27c9f60f'
    Z1RUN=[]
    for no,r in enumerate(r_list):
        with open(pathX+'Mouse'+r+'Wake_align.pickle', 'rb') as file: #w -> write; b -> binary
            outputX=pickle.load(file)
        distX=outputX['dist2']
        distXr=outputX['distr']

        # print(distXr.shape,np.mean(distXr,axis=1),np.mean(distX))
        mu=np.mean(distXr)
        sigma=np.std(np.mean(distXr,axis=1))
        zscores=(np.mean(distX)-mu)/sigma
        Z1RUN.append(zscores)
        # print(zscores)
    print(Z1RUN)

    pathX='../Data/CoreNeuron/HD/spr_corrected/'
    r_list=['12-120806','12-120807','12-120808','12-120809','12-120810','20-130517','25-140130','28-140313']
    # s='0de4b55d27c9f60f'
    Z1REM=[]
    for no,r in enumerate(r_list):
        with open(pathX+'Mouse'+r+'REM_align.pickle', 'rb') as file: #w -> write; b -> binary
            outputX=pickle.load(file)
        distX=outputX['dist2']
        distXr=outputX['distr']

        # print(distXr.shape,np.mean(distXr,axis=1),np.mean(distX))
        mu=np.mean(distXr)
        sigma=np.std(np.mean(distXr,axis=1))
        zscores=(np.mean(distX)-mu)/sigma
        Z1REM.append(zscores)
        # print(zscores)
    print(Z1REM)

    pathX='../Data/CoreNeuron/HD/spr_corrected/'
    r_list=['12-120806','12-120807','12-120808','12-120809','12-120810','20-130517','25-140130','28-140313']
    # s='0de4b55d27c9f60f'
    Z1SWS=[]
    for no,r in enumerate(r_list):
        with open(pathX+'Mouse'+r+'SWS_align.pickle', 'rb') as file: #w -> write; b -> binary
            outputX=pickle.load(file)
        distX=outputX['dist2']
        distXr=outputX['distr']

        # print(distXr.shape,np.mean(distXr,axis=1),np.mean(distX))
        mu=np.mean(distXr)
        sigma=np.std(np.mean(distXr,axis=1))
        zscores=(np.mean(distX)-mu)/sigma
        Z1SWS.append(zscores)
        # print(zscores)
    print(Z1SWS)

    data_Z1={}
    data_Z1['RUN\n' + r'$\it{HD}$']=Z1RUN
    data_Z1['REM\n' + r'$\it{HD}$']=Z1REM
    data_Z1['SWS\n' + r'$\it{HD}$']=Z1SWS
    data=[]
    for key, values in data_Z1.items():
        data.extend([(key, value) for value in values])

    cmap = ListedColormap(sns.color_palette('tab10'))
    color_dict = {'RUN\n' + r'$\it{HD}$': cmap(2), 'REM\n' + r'$\it{HD}$': cmap(4), 'SWS\n' + r'$\it{HD}$': cmap(6)} # purple, pink, blue
    
    # print(data_Z1)
    # 使用swarmplot绘制散点图
    color_palette = sns.color_palette('tab10', n_colors=len(data_Z1))
    # print([item[0] for item in data])
    print([item[1] for item in data])
    sns.swarmplot(x=[item[0] for item in data], y=[item[1] for item in data],palette=color_dict, size=5,ax=axZ1)
    axZ1.set_xticklabels(data_Z1.keys())

    print(np.array(list(data_Z1.values())))
    axZ1.bar(data_Z1.keys(),np.mean(np.array(list(data_Z1.values())),axis=1),yerr=np.std(np.array(list(data_Z1.values())),axis=1),color=color_dict.values(),alpha=0.5,zorder=5)
    axZ1.set_ylim(-8,0)


    pathX='../Data/GC1/spr/'
    s_list=['0de4b55d27c9f60f','1f20835f09e28706','5b92b96313c3fc19','7e888f1d8eaab46b','8a50a33f7fd91df4','8f7ddffaf4a5f4c5','59825ec5641c94b4','c221438d58a0b796']
    # s='0de4b55d27c9f60f'
    Z2=[]
    for no,s in enumerate(s_list):
        with open(pathX+s+'_align.pickle', 'rb') as file: #w -> write; b -> binary
            outputX=pickle.load(file)
        distX=outputX['dist2']
        distXr=outputX['distr']

        # print(distXr.shape,np.mean(distXr,axis=1),np.mean(distX))
        mu=np.mean(distXr)
        sigma=np.std(np.mean(distXr,axis=1))
        zscores=(np.mean(distX)-mu)/sigma
        Z2.append(zscores)
        # print(zscores)
    print(Z2)

    data_Z2={}
    data_Z2['OF\n' + r'$\it{GC-1}$']=Z2

    data=[]
    for key, values in data_Z2.items():
        data.extend([(key, value) for value in values])


    cmap = ListedColormap(sns.color_palette('tab20'))
    color_dict = {'OF\n' + r'$\it{GC-1}$':cmap(4),'OF\n' + r'$\it{GC-3}$': cmap(5), 'WW\n' + r'$\it{GC-3}$': cmap(2), 'REM\n' + r'$\it{GC-3}$': cmap(8),'SWS\n' + r'$\it{GC-3}$':cmap(12)} 
    
    # 使用swarmplot绘制散点图
    color_palette = sns.color_palette('tab10', n_colors=(len(data_Z2)))
    sns.swarmplot(x=[item[0] for item in data], y=[item[1] for item in data],palette=color_dict, size=5,ax=axZ3)
    # print(data_Z2.keys())
    # axZ3.set_xticklabels(data_Z2.keys()+data_Z3.keys())

    # print(list(data_Z3.values()))
    axZ3.bar(0,np.mean(np.array(list(data_Z2.values())),axis=1),yerr=np.std(np.array(list(data_Z2.values())),axis=1),color=cmap(4),alpha=0.5,zorder=5)
    
    axZ1.set_ylabel('z-score',labelpad=0.1)

    axZ3.set_xlim(-0.5,4.5)
    axZ3.set_xticklabels(['OF\n' + r'$\it{GC}$-1'])

    ax1_pos=ax1.get_position()
    axa_pos=axa.get_position()
    axc_pos=axc.get_position()
    ax71_pos=ax71.get_position()
    ax81_pos=ax81.get_position()
    axY1_pos=axY1.get_position()
    axZ1_pos=axZ1.get_position()

    plt.figtext(ax1_pos.x0-0.02,ax1_pos.y1+0.008,'A',fontsize=15)
    plt.figtext(axa_pos.x0-0.02,axa_pos.y1+0.008,'B',fontsize=15)
    plt.figtext(axc_pos.x0-0.02,axc_pos.y1+0.008,'C',fontsize=15)
    plt.figtext(ax71_pos.x0-0.02,ax71_pos.y1+0.008,'D',fontsize=15)
    plt.figtext(ax81_pos.x0-0.02,ax81_pos.y1+0.008,'E',fontsize=15)
    plt.figtext(axY1_pos.x0-0.02,axY1_pos.y1+0.002,'F',fontsize=15)
    plt.figtext(axZ1_pos.x0-0.03,axZ1_pos.y1+0.002,'G',fontsize=15)



    plt.savefig('../Figures/SuppFigure15.png',format='PNG',dpi=300)
    plt.show()
    plt.close()


def compute_cdf(data, bin_edges):
    hist, _ = np.histogram(data, bins=bin_edges)
    cdf = np.cumsum(hist) / len(data)
    return bin_edges[1:], cdf

if __name__ == '__main__':
    main()