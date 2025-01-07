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
import matplotlib.image as mpimg
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
        'labelsize' : 7.5,
}
ytick = {
        'direction' : 'out',
        'major.size' : 2,
        'major.width' : 1,
        'minor.size' : 1,
        'minor.width' : 0.5,
        'labelsize' : 7.5,
}
axes = {
        'linewidth' : 1,
        'titlesize' : 7.5,
        #'titleweight': 'bold',
        'labelsize' : 7.5,
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
    ax1=fig.add_axes([0.03,0.83,0.10*11/8.5,0.10])
    ax2=fig.add_axes([0.23,0.83,0.10*11/8.5,0.10])
    ax3=fig.add_axes([0.44,0.85,0.08,0.06])
    ax4=fig.add_axes([0.41,0.80,0.14,0.08])
    ax5=fig.add_axes([0.62,0.83,0.16,0.08])
    ax6=fig.add_axes([0.82,0.83,0.16,0.08])
    ax71=fig.add_axes([0.05,0.675,0.13,0.13*8.5/11])
    ax72=fig.add_axes([0.205,0.675,0.13,0.13*8.5/11])
    ax73=fig.add_axes([0.36,0.675,0.13,0.13*8.5/11])
    ax74=fig.add_axes([0.515,0.675,0.13,0.13*8.5/11])
    ax81=fig.add_axes([0.695,0.675,0.13,0.13*8.5/11])
    ax82=fig.add_axes([0.85,0.675,0.13,0.13*8.5/11])
    ax91=fig.add_axes([0.05,0.53,0.13,0.13*8.5/11])
    ax92=fig.add_axes([0.205,0.53,0.13,0.13*8.5/11])
    ax93=fig.add_axes([0.36,0.53,0.13,0.13*8.5/11])
    ax94=fig.add_axes([0.515,0.53,0.13,0.13*8.5/11])
    ax95=fig.add_axes([0.67,0.53,0.13,0.13*8.5/11])
    axX1=fig.add_axes([0.05,0.36,0.16,0.08])
    axX2=fig.add_axes([0.24,0.36,0.16,0.08])
    axX3=fig.add_axes([0.43,0.36,0.16,0.08])
    axY1=fig.add_axes([0.05,0.44,0.16,0.04])
    axY2=fig.add_axes([0.24,0.44,0.16,0.04])
    axY3=fig.add_axes([0.43,0.44,0.16,0.04])
    axZ1=fig.add_axes([0.64,0.36,0.12,0.12])
    # axZ2=fig.add_axes([0.77,0.36,0.04,0.12])
    axZ3=fig.add_axes([0.80,0.36,0.18,0.12])
    
    path5='../Data/GC1/spr/'
    s='1f20835f09e28706'
    with open(path5+s+'.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    Rs=output['Rs']
    d0=output['d0']
    G1=output['G1']
    pos_random=output['pos_random']  
    pos_show=output['pos_show']
    E=output['E']
    im=ax1.imshow(Rs, cmap='bwr')
    ax_pos = ax1.get_position()
    cax = fig.add_axes([ax_pos.x1 - 0.05, ax_pos.y0-0.01, 0.05, 0.005]) 
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal',ticklocation='bottom')
    cbar.ax.tick_params(labelsize=5)  
    cbar.set_ticks([np.min(Rs), np.max(Rs)])  # 设置刻度位置为最小值和最大值
    cbar.set_ticklabels(['MIN', 'MAX'])  #
    cax.tick_params(pad=0.1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Original coorrelation matrix')

    im=ax2.imshow(d0, cmap='Greys')
    ax_pos = ax2.get_position()
    cax = fig.add_axes([ax_pos.x1 - 0.05, ax_pos.y0-0.01, 0.05, 0.005]) 
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal',ticklocation='bottom')
    cbar.ax.tick_params(labelsize=5)
    cbar.set_ticks([np.min(d0), np.max(d0)])  # 设置刻度位置为最小值和最大值
    cbar.set_ticklabels(['MIN', 'MAX'])  #
    
    cax.tick_params(pad=0.1)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Phase offset matrix $\\it{D}$')

    ax_pos1 = ax1.get_position()
    ax_pos2 = ax2.get_position()

    ax1.text(38,16,'$C_{0} \\rightarrow D$', fontsize=8)
    ax1.text(38,22,'$\\rightarrow$', fontsize=25)
    ax2.text(36,22,'$\\rightarrow$', fontsize=25)


    img = mpimg.imread('../Data/compressed_spring_add.png')
    ax3.imshow(img)
    ax3_pos=ax3.get_position()
    ax3.axis('off')
    ax3.text(130, -20, r'$\Vert P_1-P_2\Vert_2<D_{12}: $compressed', fontsize=8,ha='center',va='center')
    ax3.text(150, -140, r'$E_{12}(P,D)=\frac{1}{2}k_{12}(\Vert P_1-P_2\Vert_2-D_{12})^2 $', fontsize=8,ha='center',va='center')
    img = mpimg.imread('../Data/contracted_spring_add.png')
    ax4.imshow(img)
    ax4.axis('off')
    ax4.text(200, 10, r'$\Vert P_1-P_2\Vert_2>D_{12}: $stretched', fontsize=8,ha='center',va='center')

    ax4.text(401,-34,'$\\rightarrow$', fontsize=25)

    ax6.plot(np.arange(len(E)), E, c='k')
    ax6.spines['right'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    ax6.set_xlabel('Steps',labelpad=0.1)
    ax6.set_ylabel(r'$E$',labelpad=0.1)
    ax6.set_title('                      $GC$-1',fontsize=9)

    ax6_pos = ax6.get_position()
    ax61 = fig.add_axes([ax6_pos.x0+0.015, ax6_pos.y1 - 0.024, 0.05, 0.04])
    ax61.set_title('Initial\nrandom layout',pad=0.1)
    nx.draw_networkx(G1, node_size=20, node_color='tab:red', pos=pos_random, edgecolors='none', edge_color='grey', width=0.1, with_labels=False)
    ax61.axis('off')
    ax62 = fig.add_axes([ax6_pos.x1-0.06, ax6_pos.y0 +0.015, 0.05, 0.04])
    ax62.set_title('Final layout',pad=0.1)
    nx.draw_networkx(G1, node_size=20, node_color='tab:blue', pos=pos_show, edgecolors='none', edge_color='grey', width=0.1, with_labels=False)
    ax62.axis('off')

    path3='../Data/HD/ProcessedData/spr/'
    m='Mouse12-120806'
    s='REM'
    with open(path3+m+'_'+s+'.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    HDRs=output['HDRs']   
    d0=output['d0']  
    G1=output['G1']  
    pos_random=output['pos_random']  
    pos_show=output['pos_show']
    E=output['E']  
    ax5.plot(np.arange(len(E)), E, c='k')
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    ax5.set_xlabel('Steps',labelpad=0.1)
    ax5.set_ylabel(r'$E$',labelpad=0.1)
    ax5.set_title('                          $HD$',fontsize=9)

    ax_pos = ax5.get_position()  
    # plt.figtext(ax_pos.x0-0.03,ax_pos.y1+0.02,'c',fontsize=abc_size)

    ax5_pos = ax5.get_position()
    ax51 = fig.add_axes([ax5_pos.x0+0.015, ax5_pos.y1 - 0.024, 0.05, 0.04])
    ax51.set_title('Initial\nrandom layout',pad=0.1)
    nx.draw_networkx(G1, node_size=20, node_color='tab:red', pos=pos_random, edgecolors='none', edge_color='grey', width=0.1, with_labels=False)
    ax51.axis('off')
    ax52 = fig.add_axes([ax5_pos.x1-0.06, ax5_pos.y0 +0.015, 0.05, 0.04])
    ax52.set_title('Final layout',pad=0.1)
    nx.draw_networkx(G1, node_size=20, node_color='tab:blue', pos=pos_show, edgecolors='none', edge_color='grey', width=0.1, with_labels=False)
    ax52.axis('off')

    path7='../Data/HD/spr_corrected/'#'../Data/HD/ProcessedData/spr/'
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
    
    # Draw grey circle
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
    ax71.tick_params(pad=1)
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
    ax72.tick_params(pad=1)
    ax73.set_xlim(0,2*np.pi)
    ax73.set_xticks([0,np.pi,2*np.pi])
    ax73.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax73.set_ylim(0,2*np.pi)
    ax73.set_yticks([0,np.pi,2*np.pi])
    ax73.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax73.set_xlabel('Actual $\\theta$ (rad)',labelpad=0.1)
    # ax73.set_ylabel('Estimated $\\theta$ (rad)',labelpad=0.1)
    ax73.tick_params(pad=1)
    ax74.set_xlim(0,2*np.pi)
    ax74.set_xticks([0,np.pi,2*np.pi])
    ax74.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax74.set_ylim(0,2*np.pi)
    ax74.set_yticks([0,np.pi,2*np.pi])
    ax74.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax74.set_xlabel('Actual $\\theta$ (rad)',labelpad=0.1)
    # ax74.set_ylabel('Estimated $\\theta$ (rad)',labelpad=0.1)
    ax74.tick_params(pad=1)

    # Draw lines between pos1 and aligned_pos2
    # for n in range(len(pos71a)):
    #     ax71.plot([pos71a[n, 0], pos71b[n, 0]], [pos71a[n, 1], pos71b[n, 1]], c='grey', linewidth=0.6)
    # for n in range(len(pos72a)):
    #     ax72.plot([pos72a[n, 0], pos72b[n, 0]], [pos72a[n, 1], pos72b[n, 1]], c='grey', linewidth=0.6)
    # for n in range(len(pos73a)):
    #     ax73.plot([pos73a[n, 0], pos73b[n, 0]], [pos73a[n, 1], pos73b[n, 1]], c='grey', linewidth=0.6)
    # for n in range(len(pos71a)):
    #     ax74.plot([pos71a[n, 0], pos_random[n, 0]], [pos71a[n, 1], pos_random[n, 1]], c='grey', linewidth=0.6)
    
    # ax71.set_aspect('equal', 'box')
    # ax71.axis('off')
    # ax72.axis('off')
    # ax73.axis('off')
    # ax74.axis('off')
    ax71.set_title(r'$\it{HD}$ '+'RUN', fontsize=8,pad=0.1)
    ax72.set_title(r'$\it{HD}$ '+'REM', fontsize=8,pad=0.1)
    ax73.set_title(r'$\it{HD}$ '+'SWS', fontsize=8,pad=0.1)
    ax74.set_title(r'$\it{HD}$ '+'Random', fontsize=8,pad=0.1)

    path9='../Data/GC1/spr/'
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
    print(pos81b)
    # plot_phase_distribution(pos81a, 'darkgray', pos81b, 'tab:blue', ax81, s=12)
    # ax81.set_title(s, fontsize=8)

    ax81.scatter(pos81a[:,0],pos81b[:,0],marker='o',edgecolor='none',facecolor='tab:blue',alpha=0.5)
    ax81.scatter(pos81a[:,1],pos81b[:,1],marker='o',edgecolor='tab:blue',facecolor='none',alpha=0.5)
    
    ax81.set_title(r'$\it{GC}$-1 '+'OF', fontsize=8,pad=0.1)

    # Right subplot
    # plot_phase_distribution(pos81a, 'darkgray', posrs[0,:,:], 'tab:red', ax82, s=12)
    ax82.set_title(r'$\it{GC}$-1 '+'Random', fontsize=8,pad=0.1)
    
    ax82.scatter(pos81a[:,0],posrs[0,:,0],marker='o',edgecolor='none',facecolor='tab:red',alpha=0.5)
    ax82.scatter(pos81a[:,1],posrs[0,:,1],marker='o',edgecolor='tab:red',facecolor='none',alpha=0.5)
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
    ax81.tick_params(pad=1)
    ax82.tick_params(pad=1)
    


    path8='../Data/GC3/spr/'
    m='Q_2_'
    states=['OF','WW','REM','SWS']
    with open(path8+m+'OF'+'_align.pickle', 'rb') as file: #w -> write; b -> binary
        output91=pickle.load(file)
        pos91a=output91['pos1']
        pos91b=output91['pos2']
        dist91=output91['dist2']
    with open(path8+m+'WW'+'_align.pickle', 'rb') as file: #w -> write; b -> binary
        output92=pickle.load(file)
        pos92a=output92['pos1']
        pos92b=output92['pos2']
        dist92=output92['dist2']
    with open(path8+m+'REM'+'_align.pickle', 'rb') as file: #w -> write; b -> binary
        output93=pickle.load(file)
        pos93a=output93['pos1']
        pos93b=output93['pos2']
        dist93=output93['dist2']
    with open(path8+m+'SWS'+'_align.pickle', 'rb') as file: #w -> write; b -> binary
        output94=pickle.load(file)
        pos94a=output94['pos1']
        pos94b=output94['pos2']
        dist94=output94['dist2']
    
    posrs=output91['posrs']
    pos_random=posrs[0,:,:]
    dist9r=output91['distr']

    pos91a += np.pi
    pos91b += np.pi
    pos91a = np.clip(pos91a, 0, 2*np.pi)
    pos91b = np.clip(pos91b, 0, 2*np.pi)
    # plot_phase_distribution(pos91a, 'darkgray', pos91b, 'tab:blue', ax91, s=12)
    
    ax91.scatter(pos91a[:,0],pos91b[:,0],marker='o',edgecolor='none',facecolor='tab:blue',alpha=0.5)
    ax91.scatter(pos91a[:,1],pos91b[:,1],marker='o',edgecolor='tab:blue',facecolor='none',alpha=0.5)

    ax91.set_title(r'$\it{GC}$-3 '+'OF', fontsize=8,pad=0.1)
    
    pos92a += np.pi
    pos92b += np.pi
    pos92a = np.clip(pos92a, 0, 2*np.pi)
    pos92b = np.clip(pos92b, 0, 2*np.pi)
    # plot_phase_distribution(pos92a, 'darkgray', pos92b, 'tab:orange', ax92, s=12)
    
    ax92.scatter(pos92a[:,0],pos92b[:,0],marker='o',edgecolor='none',facecolor='tab:orange',alpha=0.5)
    ax92.scatter(pos92a[:,1],pos92b[:,1],marker='o',edgecolor='tab:orange',facecolor='none',alpha=0.5)

    ax92.set_title(r'$\it{GC}$-3 '+'WW', fontsize=8,pad=0.1)
    
    pos93a += np.pi
    pos93b += np.pi
    pos93a = np.clip(pos93a, 0, 2*np.pi)
    pos93b = np.clip(pos93b, 0, 2*np.pi)
    # plot_phase_distribution(pos93a, 'darkgray', pos93b, 'tab:green', ax93, s=12)
    
    ax93.scatter(pos93a[:,0],pos93b[:,0],marker='o',edgecolor='none',facecolor='tab:green',alpha=0.5)
    ax93.scatter(pos93a[:,1],pos93b[:,1],marker='o',edgecolor='tab:green',facecolor='none',alpha=0.5)

    ax93.set_title(r'$\it{GC}$-3 '+'REM', fontsize=8,pad=0.1)
    
    pos94a += np.pi
    pos94b += np.pi
    pos94a = np.clip(pos94a, 0, 2*np.pi)
    pos94b = np.clip(pos94b, 0, 2*np.pi)
    # plot_phase_distribution(pos94a, 'darkgray', pos94b, 'tab:purple', ax94, s=12)
    
    ax94.scatter(pos94a[:,0],pos94b[:,0],marker='o',edgecolor='none',facecolor='tab:purple',alpha=0.5)
    ax94.scatter(pos94a[:,1],pos94b[:,1],marker='o',edgecolor='tab:purple',facecolor='none',alpha=0.5)

    ax94.set_title(r'$\it{GC}$-3 '+'SWS', fontsize=8,pad=0.1)
    
    # plot_phase_distribution(pos91a, 'darkgray', posrs[0,:,:], 'tab:red', ax95, s=12)
    
    ax95.scatter(pos91a[:,0],posrs[0,:,0],marker='o',edgecolor='none',facecolor='tab:red',alpha=0.5)
    ax95.scatter(pos91a[:,1],posrs[0,:,1],marker='o',edgecolor='tab:red',facecolor='none',alpha=0.5)

    ax95.set_title(r'$\it{GC}$-3 '+'Random', fontsize=8,pad=0.1)

    ax91.tick_params(pad=1)
    ax92.tick_params(pad=1)
    ax93.tick_params(pad=1)
    ax94.tick_params(pad=1)
    ax95.tick_params(pad=1)
    
    ax91.set_xlim(0,2*np.pi)
    ax91.set_xticks([0,np.pi,2*np.pi])
    ax91.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax91.set_ylim(0,2*np.pi)
    ax91.set_yticks([0,np.pi,2*np.pi])
    ax91.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax91.set_xlabel('Actual $g_1$ or $g_2$ (rad)',labelpad=0.1)
    ax91.set_ylabel('Estimated $g_1$ or $g_2$ (rad)',labelpad=0.1)
    ax92.set_xlim(0,2*np.pi)
    ax92.set_xticks([0,np.pi,2*np.pi])
    ax92.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax92.set_ylim(0,2*np.pi)
    ax92.set_yticks([0,np.pi,2*np.pi])
    ax92.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax92.set_xlabel('Actual $g_1$ or $g_2$ (rad)',labelpad=0.1)
    # ax92.set_ylabel('Estimated $g_1$ or $g_2$ (rad)',labelpad=0.1)
    ax93.set_xlim(0,2*np.pi)
    ax93.set_xticks([0,np.pi,2*np.pi])
    ax93.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax93.set_ylim(0,2*np.pi)
    ax93.set_yticks([0,np.pi,2*np.pi])
    ax93.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax93.set_xlabel('Actual $g_1$ or $g_2$ (rad)',labelpad=0.1)
    # ax93.set_ylabel('Estimated $g_1$ or $g_2$ (rad)',labelpad=0.1)
    ax94.set_xlim(0,2*np.pi)
    ax94.set_xticks([0,np.pi,2*np.pi])
    ax94.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax94.set_ylim(0,2*np.pi)
    ax94.set_yticks([0,np.pi,2*np.pi])
    ax94.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax94.set_xlabel('Actual $g_1$ or $g_2$ (rad)',labelpad=0.1)
    # ax94.set_ylabel('Estimated $g_1$ or $g_2$ (rad)',labelpad=0.1)
    ax95.set_xlim(0,2*np.pi)
    ax95.set_xticks([0,np.pi,2*np.pi])
    ax95.set_xticklabels([0,'$\\pi$','$2\\pi$'])
    ax95.set_ylim(0,2*np.pi)
    ax95.set_yticks([0,np.pi,2*np.pi])
    ax95.set_yticklabels([0,'$\\pi$','$2\\pi$'])
    ax95.set_xlabel('Actual $g_1$ or $g_2$ (rad)',labelpad=0.1)
    # ax95.set_ylabel('Estimated $g_1$ or $g_2$ (rad)',labelpad=0.1)

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
    axY2.vlines(np.mean(dist8r,axis=1),0,1,color='tab:red',alpha=0.2)
    axY2.vlines(np.mean(dist81),0,2,color='tab:blue')

    axX2.set_xlim(-0.1,4.5)
    axY2.set_xlim(-0.1,4.5)
    axX2.set_xticks([0,1,2,3,4])
    axY2.set_xticks([0,1,2,3,4],[])
    axX2.set_ylim(0,1.15)
    axY2.set_ylim(0,3.5)
    axX2.set_yticks([0,0.5,1.0])
    axY2.set_yticks([0,1,2,3])
    axX2.set_xlabel('Estimation error (rad)',labelpad=0.1)
    # axX2.set_ylabel('Cumulative fraction',labelpad=0.1)
    axY2.set_ylabel(None)

    min_value = 0
    max_value = max(dist91.max(), dist9r.max())
    # 将范围划分为相等的间隔
    num_bins=500
    bin_edges = np.linspace(min_value, max_value, num_bins + 1)
    
    for t in range(dist9r.shape[0]):   
        # print(t)
        x2, y2 = compute_cdf(dist9r[t, :], bin_edges)
        if y2[0] != 0:
            x2 = np.insert(x2, 0, 0)
            y2 = np.insert(y2, 0, 0)        
        
        axX3.plot(x2, y2, c='tab:red',alpha=0.2)
    # 计算 dist1 和 dist2 的 CDF
    x1, y1 = compute_cdf(dist91, bin_edges)      
    if y1[0] != 0:
        x1 = np.insert(x1, 0, 0)
        y1 = np.insert(y1, 0, 0)
    axX3.plot(x1, y1, label='dist1', c='tab:blue') 
    x1, y1 = compute_cdf(dist92, bin_edges)      
    if y1[0] != 0:
        x1 = np.insert(x1, 0, 0)
        y1 = np.insert(y1, 0, 0)
    axX3.plot(x1, y1, label='dist1', c='tab:orange') 
    x1, y1 = compute_cdf(dist93, bin_edges)      
    if y1[0] != 0:
        x1 = np.insert(x1, 0, 0)
        y1 = np.insert(y1, 0, 0)
    axX3.plot(x1, y1, label='dist1', c='tab:green') 
    x1, y1 = compute_cdf(dist94, bin_edges)      
    if y1[0] != 0:
        x1 = np.insert(x1, 0, 0)
        y1 = np.insert(y1, 0, 0)
    axX3.plot(x1, y1, label='dist1', c='tab:purple') 
    
    sns.kdeplot(data=np.mean(dist9r,axis=1),ax=axY3,color='tab:red')
    axY3.vlines(np.mean(dist9r,axis=1),0,1,color='tab:red',alpha=0.2)
    axY3.vlines(np.mean(dist91),0,2, color='tab:blue')
    axY3.vlines(np.mean(dist92),0,2, color='tab:orange')
    axY3.vlines(np.mean(dist93),0,2, color='tab:green')
    axY3.vlines(np.mean(dist94),0,2, color='tab:purple')


    axX3.set_xlim(-0.1,4.5)
    axY3.set_xlim(-0.1,4.5)
    axX3.set_xticks([0,1,2,3,4])
    axY3.set_xticks([0,1,2,3,4],[])
    axX3.set_ylim(0,1.15)
    axY3.set_ylim(0,3.5)
    axX3.set_yticks([0,0.5,1.0])
    axY3.set_yticks([0,1,2,3])
    axX3.set_xlabel('Estimation error (rad)',labelpad=0.1)
    # axX3.set_ylabel('Cumulative fraction',labelpad=0.1)
    axY3.set_ylabel(None)

    pathX='../Data/HD/spr/'
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

    pathX='../Data/HD/spr/'
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

    pathX='../Data/HD/spr/'
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
    
    pathX='../Data/GC3/spr/'
    s_list=['Q_1_OF','Q_2_OF','R_1_OF_day1','R_1_OF_day2','R_2_OF_day1','R_2_OF_day2','R_3_OF_day1','S_1_OF']
    # s='0de4b55d27c9f60f'
    Z3OF=[]
    for no,s in enumerate(s_list):
        with open(pathX+s+'_align.pickle', 'rb') as file: #w -> write; b -> binary
            outputX=pickle.load(file)
        distX=outputX['dist2']
        distXr=outputX['distr']

        # print(distXr.shape,np.mean(distXr,axis=1),np.mean(distX))
        mu=np.mean(distXr)
        sigma=np.std(np.mean(distXr,axis=1))
        zscores=(np.mean(distX)-mu)/sigma
        Z3OF.append(zscores)
        # print(zscores)
    print(Z3OF)

    pathX='../Data/GC3/spr/'
    s_list=['Q_1_WW','Q_2_WW','R_1_WW_day1','R_2_WW_day1','R_3_WW_day1','S_1_WW',]
    # s='0de4b55d27c9f60f'
    Z3WW=[]
    for no,s in enumerate(s_list):
        with open(pathX+s+'_align.pickle', 'rb') as file: #w -> write; b -> binary
            outputX=pickle.load(file)
        distX=outputX['dist2']
        distXr=outputX['distr']

        # print(distXr.shape,np.mean(distXr,axis=1),np.mean(distX))
        mu=np.mean(distXr)
        sigma=np.std(np.mean(distXr,axis=1))
        zscores=(np.mean(distX)-mu)/sigma
        Z3WW.append(zscores)
        # print(zscores)
    print(Z3WW)

    pathX='../Data/GC3/spr/'
    s_list=['Q_1_REM','Q_2_REM','R_1_REM_day2','R_2_REM_day2','S_1_REM']
    # s='0de4b55d27c9f60f'
    Z3REM=[]
    for no,s in enumerate(s_list):
        with open(pathX+s+'_align.pickle', 'rb') as file: #w -> write; b -> binary
            outputX=pickle.load(file)
        distX=outputX['dist2']
        distXr=outputX['distr']

        # print(distXr.shape,np.mean(distXr,axis=1),np.mean(distX))
        mu=np.mean(distXr)
        sigma=np.std(np.mean(distXr,axis=1))
        zscores=(np.mean(distX)-mu)/sigma
        Z3REM.append(zscores)
        # print(zscores)
    print(Z3REM)

    pathX='../Data/GC3/spr/'
    s_list=['Q_1_SWS','Q_2_SWS','R_1_SWS_day2','R_2_SWS_day2','S_1_SWS']
    # s='0de4b55d27c9f60f'
    Z3SWS=[]
    for no,s in enumerate(s_list):
        with open(pathX+s+'_align.pickle', 'rb') as file: #w -> write; b -> binary
            outputX=pickle.load(file)
        distX=outputX['dist2']
        distXr=outputX['distr']

        # print(distXr.shape,np.mean(distXr,axis=1),np.mean(distX))
        mu=np.mean(distXr)
        sigma=np.std(np.mean(distXr,axis=1))
        zscores=(np.mean(distX)-mu)/sigma
        Z3SWS.append(zscores)
        # print(zscores)
    print(Z3SWS)


    data_Z3={}
    data_Z3['OF\n' + r'$\it{GC-3}$']=Z3OF
    data_Z3['WW\n' + r'$\it{GC-3}$']=Z3WW
    data_Z3['REM\n' + r'$\it{GC-3}$']=Z3REM
    data_Z3['SWS\n' + r'$\it{GC-3}$']=Z3SWS

    data=[]
    for key, values in data_Z2.items():
        data.extend([(key, value) for value in values])
    for key, values in data_Z3.items():
        data.extend([(key, value) for value in values])

    cmap = ListedColormap(sns.color_palette('tab20'))
    color_dict = {'OF\n' + r'$\it{GC-1}$':cmap(4),'OF\n' + r'$\it{GC-3}$': cmap(5), 'WW\n' + r'$\it{GC-3}$': cmap(2), 'REM\n' + r'$\it{GC-3}$': cmap(8),'SWS\n' + r'$\it{GC-3}$':cmap(12)} 
    
    # 使用swarmplot绘制散点图
    color_palette = sns.color_palette('tab10', n_colors=(len(data_Z2)+len(data_Z3)))
    sns.swarmplot(x=[item[0] for item in data], y=[item[1] for item in data],palette=color_dict, size=5,ax=axZ3)
    # print(data_Z2.keys())
    # axZ3.set_xticklabels(data_Z2.keys()+data_Z3.keys())

    # print(list(data_Z3.values()))
    axZ3.bar(0,np.mean(np.array(list(data_Z2.values())),axis=1),yerr=np.std(np.array(list(data_Z2.values())),axis=1),color=cmap(4),alpha=0.5,zorder=5)
    axZ3.bar(1,np.mean(data_Z3['OF\n' + r'$\it{GC-3}$']),yerr=np.std(data_Z3['OF\n' + r'$\it{GC-3}$']),color=cmap(5),alpha=0.5,zorder=5)
    axZ3.bar(2,np.mean(data_Z3['WW\n' + r'$\it{GC-3}$']),yerr=np.std(data_Z3['WW\n' + r'$\it{GC-3}$']),color=cmap(2),alpha=0.5,zorder=5)
    axZ3.bar(3,np.mean(data_Z3['REM\n' + r'$\it{GC-3}$']),yerr=np.std(data_Z3['REM\n' + r'$\it{GC-3}$']),color=cmap(8),alpha=0.5,zorder=5)
    axZ3.bar(4,np.mean(data_Z3['SWS\n' + r'$\it{GC-3}$']),yerr=np.std(data_Z3['SWS\n' + r'$\it{GC-3}$']),color=cmap(12),alpha=0.5,zorder=5)
    
    axZ1.set_ylabel('z-score',labelpad=0.1)
    axZ3.set_xticklabels(['OF\n' + r'$\it{GC}$-1',
                        'OF\n' + r'$\it{GC}$-3',
                        'WW\n' + r'$\it{GC}$-3',
                        'REM\n' + r'$\it{GC}$-3',
                        'SWS\n' + r'$\it{GC}$-3',
                        ])

    ax1_pos=ax1.get_position()
    ax71_pos=ax71.get_position()
    ax81_pos=ax81.get_position()
    ax91_pos=ax91.get_position()
    axY1_pos=axY1.get_position()
    axZ1_pos=axZ1.get_position()

    plt.figtext(ax1_pos.x0-0.025,ax1_pos.y1+0.008,'A',fontsize=15)
    plt.figtext(ax1_pos.x0-0.025,ax71_pos.y1+0.008,'B',fontsize=15)
    plt.figtext(ax81_pos.x0-0.02,ax81_pos.y1+0.008,'C',fontsize=15)
    plt.figtext(ax1_pos.x0-0.025,ax91_pos.y1+0.008,'D',fontsize=15)
    plt.figtext(ax1_pos.x0-0.025,axY1_pos.y1+0.005,'E',fontsize=15)
    plt.figtext(axZ1_pos.x0-0.02,axZ1_pos.y1+0.005,'F',fontsize=15)



    plt.savefig('../Figures/Figure5.png',format='PNG',dpi=300)
    plt.show()
    plt.close()


def compute_cdf(data, bin_edges):
    hist, _ = np.histogram(data, bins=bin_edges)
    cdf = np.cumsum(hist) / len(data)
    return bin_edges[1:], cdf

if __name__ == '__main__':
    main()