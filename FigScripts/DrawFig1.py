# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 2023

@author: Tao WANG

Description: Draw Fig1 of RTO
"""

####加载库####
import numpy as np
np.random.seed(20231230)
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colors
import random
import networkx as nx
import sys
sys.path.append('../') # 将上级目录加入 sys.path
from DrawStandard import *
import pickle
# from DetectCP import GenerateGraph, CalCoreness, Metrics, DrawNetwork
import seaborn as sns

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
    # 3个HD+3个nonHD神经元放电的展示
    ax11_x=0.05
    ax13=fig.add_axes([ax11_x,0.75+0.06*0,0.05,0.05],polar=True)
    ax12=fig.add_axes([ax11_x,0.75+0.06*1,0.05,0.05],polar=True)
    ax11=fig.add_axes([ax11_x,0.75+0.06*2,0.05,0.05],polar=True)
    ax_pos11 = ax11.get_position()
    ax16=fig.add_axes([ax11_x+0.06,0.75+0.06*0,0.05,0.05],polar=True)
    ax15=fig.add_axes([ax11_x+0.06,0.75+0.06*1,0.05,0.05],polar=True)
    ax14=fig.add_axes([ax11_x+0.06,0.75+0.06*2,0.05,0.05],polar=True)
    # cell pair correlation分布的展示
    ax2_y=0.79
    ax21_x=0.21
    ax21=fig.add_axes([ax21_x,ax2_y,0.08,0.06])
    ax22=fig.add_axes([ax21_x+0.1,ax2_y,0.08,0.06])
    ax23=fig.add_axes([ax21_x+0.2,ax2_y,0.08,0.06])
    # 3个grid+3个nongrid神经元放电的展示
    ax31_x=0.51
    ax33=fig.add_axes([ax31_x,0.76+0.06*0,0.05,0.05])
    ax32=fig.add_axes([ax31_x,0.76+0.06*1,0.05,0.05])
    ax31=fig.add_axes([ax31_x,0.76+0.06*2,0.05,0.05])
    ax_pos31 = ax31.get_position()
    ax36=fig.add_axes([ax31_x+0.065,0.76+0.06*0,0.05,0.05])
    ax35=fig.add_axes([ax31_x+0.065,0.76+0.06*1,0.05,0.05])
    ax34=fig.add_axes([ax31_x+0.065,0.76+0.06*2,0.05,0.05])
    # cell pair correlation分布的展示
    ax41=fig.add_axes([ax31_x+0.17,ax2_y,0.08,0.06])
    ax42=fig.add_axes([ax31_x+0.27,ax2_y,0.08,0.06])
    ax43=fig.add_axes([ax31_x+0.37,ax2_y,0.08,0.06])
    # cell pair correlation的imshow
    # wi=0.03
    ax51_x0=0.05
    ax51_wi=0.19
    ax51_w=0.12
    ax51=fig.add_axes([0.05,0.62,ax51_w,0.1])
    ax_pos51 = ax51.get_position()
    ax52=fig.add_axes([0.24,0.62,ax51_w,0.1])
    ax53=fig.add_axes([0.43,0.62,ax51_w,0.1])
    ax54=fig.add_axes([0.64,0.62,ax51_w,0.1])
    ax_pos54 = ax54.get_position()
    ax55=fig.add_axes([0.84,0.62,ax51_w,0.1])
    # cell pair correlation的kdeplot
    ax61_y=0.47
    ax61=fig.add_axes([0.05,ax61_y,0.15,0.1])
    ax_pos61 = ax61.get_position()
    ax62=fig.add_axes([0.23,ax61_y,0.15,0.1])
    ax63=fig.add_axes([0.41,ax61_y,0.15,0.1])
    ax64=fig.add_axes([0.63,ax61_y,0.15,0.1])
    ax65=fig.add_axes([0.81,ax61_y,0.15,0.1])
    ax_pos64 = ax64.get_position()
    # core-periphery structure 的展示
    ax71=fig.add_axes([0.05,0.33,0.10,0.10/11*8.5])
    ax_pos71 = ax71.get_position()
    ax72=fig.add_axes([0.20,0.33,0.10,0.10/11*8.5])
    ax73=fig.add_axes([0.33,0.32,0.12,0.10])
    ax74=fig.add_axes([0.46,0.33,0.10,0.10/11*8.5])
    # 证明cp结构的存在
    ax81=fig.add_axes([0.60,0.33,0.10,0.10/11*8.5])
    ax82=fig.add_axes([0.73,0.33,0.10,0.10/11*8.5])
    ax83=fig.add_axes([0.86,0.33,0.10,0.10/11*8.5])
    ax_pos81 = ax81.get_position()

    # Fig1a HD cell pair
    # Code_readme/HD/cellpair_example.py 
    path1a='../Data/HD/ProcessedData/cellpair_example/'
    with open(path1a+'output.pickle', 'rb') as file: #w -> write; b -> binary
        output = pickle.load(file)
    Orienedges = output['Orienedges']
    RateHist = output['RateHist']
    Rs = output['Rs']
    axes = [ax11, ax12, ax13, ax14, ax15, ax16]
    maxfs = ['13.3', '8.4', '3.4', '10.5', '13.1', '5.0']
    for no, ax in enumerate(axes):
        # Normalize the rate
        normalized_rate = RateHist[no] / np.nanmax(RateHist[no])
        # Plot the data
        if no<3:
            ax.plot((Orienedges[1:] + Orienedges[:-1]) / 2, normalized_rate, color='b', linewidth=2.0) # dark pink
        else:
            ax.plot((Orienedges[1:] + Orienedges[:-1]) / 2, normalized_rate, color='grey', linewidth=2.0) # dark pink
        ax.plot([0,-np.pi],[1,1],linewidth=1.0,color='k')
        ax.plot([np.pi/2,3*np.pi/2],[1,1],linewidth=1.0,color='k')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_rgrids([0.3,0.7], angle=45.0)
        ax.axis('off')
        # ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315])
        
        if no<3:
            ax.set_title('H$_{%d}$         '%(no+1),fontsize=9,pad=0.1)
        else:
            ax.set_title('N$_{%d}$         '%(no-2),fontsize=9,pad=0.1)

    ts=np.arange(-5, 5+1e-3, 0.01)
    ax21.plot(ts, Rs[0,:], label='H$_1$-H$_2$', color='b',alpha=1.0)
    ax21.plot(ts, Rs[1,:], label='H$_1$-H$_3$', color='b',alpha=0.6)
    ax21.plot(ts, Rs[2,:], label='H$_2$-H$_3$', color='b',alpha=0.2)
    ax21.set_xlim(-5,5)
    ax21.set_ylim(-0.25,0.5)
    ax21.set_yticks([-0.2,0,0.2,0.4])
    ax21.set_xlabel('Lag (s)')
    ax21.set_ylabel('Correlation coefficient', labelpad=0.1)
    ax21.legend(bbox_to_anchor=(0.04, 1.6, 0.2, 0.2), loc='upper left')
    
    ax22.plot(ts, Rs[7,:], color='grey',alpha=0.5)
    ax22.plot(ts, Rs[8,:], color='grey',alpha=0.5)
    ax22.plot(ts, Rs[9,:], color='grey',alpha=0.5)
    ax22.plot(ts, Rs[10,:], color='grey',alpha=0.5)
    ax22.plot(ts, Rs[11,:], color='grey',alpha=0.5)
    ax22.plot(ts, Rs[3,:], label='H$_1$-N$_1$', color='tab:orange',alpha=1.0)
    ax22.plot(ts, Rs[4,:], label='H$_1$-N$_2$', color='tab:orange',alpha=0.7)
    ax22.plot(ts, Rs[5,:], label='H$_1$-N$_3$', color='tab:orange',alpha=0.4)
    ax22.plot(ts, Rs[6,:], label='others', color='grey',alpha=0.5)
    ax22.set_xlim(-5,5)
    ax22.set_ylim(-0.25,0.5)
    ax22.set_yticks([-0.2,0,0.2,0.4])
    ax22.yaxis.set_ticklabels([])
    ax22.yaxis.set_ticks_position('left')
    ax22.set_xlabel('Lag (s)')
    ax22.legend(bbox_to_anchor=(0.04, 1.8, 0.2, 0.2), loc='upper left')

    ax23.plot(ts, Rs[12,:], label='N$_1$-N$_2$', color='grey',alpha=1.0)
    ax23.plot(ts, Rs[13,:], label='N$_1$-N$_3$', color='grey',alpha=0.6)
    ax23.plot(ts, Rs[14,:], label='N$_2$-N$_3$', color='grey',alpha=0.2)
    ax23.set_xlim(-5,5)
    ax23.set_ylim(-0.25,0.5)
    ax23.set_yticks([-0.2,0,0.2,0.4])
    ax23.yaxis.set_ticklabels([])
    ax23.yaxis.set_ticks_position('left')
    ax23.set_xlabel('Lag (s)')
    ax23.legend(bbox_to_anchor=(0.04, 1.6, 0.2, 0.2), loc='upper left')

    # Fig1b grid cell pair
    # Code_readme/GC1/cellpair_example.py 
    path1b='../Data/GC1/cellpair_example/'
    with open(path1b+'output.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    ratemap=output['ratemap']
    Rs=output['Rs']
    axes=[ax31, ax32, ax33, ax34, ax35, ax36]
    for no, ax in enumerate(axes):
        normalized_rate = ratemap[no,:,:] / np.nanmax(ratemap[no,:,:])
        ax.set_xticks([])
        ax.set_yticks([])
        ax_pos = ax.get_position()  
        if no<3:
            im=ax.imshow(normalized_rate, cmap='jet', vmax=0.85*np.max(normalized_rate), vmin=0)
            ax.set_title('G$_%d$         '%(no+1),fontsize=9,pad=0.1)
            # if no==2:
            #     cax = fig.add_axes([ax_pos.x0+0.01, ax_pos.y0 - 0.004, 0.04, 0.004])
            #     cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            #     cbar.ax.tick_params(labelsize=5)  
            #     cbar.set_ticks([0, 0.85])  # 设置刻度位置为最小值和最大值
            #     cbar.set_ticklabels(['MIN', 'MAX'])  #

        else:
            im=ax.imshow(normalized_rate, cmap='jet', vmax=0.85*np.max(normalized_rate), vmin=0)
            ax.set_title('N$_%d$         '%(no-2),fontsize=9,pad=0.1)
            if no==5:
                cax = fig.add_axes([ax_pos.x0+0.01, ax_pos.y0 - 0.004, 0.04, 0.004])
                cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
                cbar.ax.tick_params(labelsize=5)  
                cbar.set_ticks([0, 0.85])  # 设置刻度位置为最小值和最大值
                cbar.set_ticklabels(['MIN', 'MAX'])  #

    ts=np.arange(-10, 10+1e-3, 0.2)
    ax41.plot(ts, Rs[0,:], label='G$_1$-G$_2$', color='b',alpha=1.0)
    ax41.plot(ts, Rs[1,:], label='G$_1$-G$_3$', color='b',alpha=0.6)
    ax41.plot(ts, Rs[2,:], label='G$_2$-G$_3$', color='b',alpha=0.2)
    ax41.set_xlim(-10,10)
    ax41.set_ylim(-0.2,0.6)
    ax41.set_yticks([-0.2,0,0.2,0.4,0.6])
    ax41.set_yticklabels([-0.2,0,0.2,0.4,0.6])
    ax41.set_xlabel('Lag (s)')
    ax41.set_ylabel('Correlation coefficient', labelpad=0.1)
    ax41.legend(bbox_to_anchor=(0.04, 1.6, 0.2, 0.2), loc='upper left')
    
    ax42.plot(ts, Rs[6,:], color='grey',alpha=0.2)
    ax42.plot(ts, Rs[7,:], color='grey',alpha=0.2)
    ax42.plot(ts, Rs[9,:], color='grey',alpha=0.2)
    ax42.plot(ts, Rs[10,:], color='grey',alpha=0.2)
    ax42.plot(ts, Rs[11,:], color='grey',alpha=0.2)
    ax42.plot(ts, Rs[3,:], label='G$_1$-N$_1$', color='tab:orange',alpha=1.0)
    ax42.plot(ts, Rs[4,:], label='G$_1$-N$_2$', color='tab:orange',alpha=0.7)
    ax42.plot(ts, Rs[5,:], label='G$_1$-N$_3$', color='tab:orange',alpha=0.4)
    ax42.plot(ts, Rs[8,:], color='grey',alpha=0.2, label='others')

    ax42.set_xlim(-10,10)
    ax42.set_ylim(-0.2,0.6)
    ax42.set_yticks([-0.2,0,0.2,0.4,0.6])
    ax42.yaxis.set_ticklabels([])
    ax42.yaxis.set_ticks_position('left')
    ax42.set_xlabel('Lag (s)')
    ax42.legend(bbox_to_anchor=(0.04, 1.8, 0.2, 0.2), loc='upper left')

    ax43.plot(ts, Rs[12,:], label='N$_1$-N$_2$', color='grey',alpha=1.0)
    ax43.plot(ts, Rs[13,:], label='N$_1$-N$_3$', color='grey',alpha=0.6)
    ax43.plot(ts, Rs[14,:], label='N$_2$-N$_3$', color='grey',alpha=0.2)
    ax43.set_xlim(-10,10)
    ax43.set_ylim(-0.2,0.6)
    ax43.set_yticks([-0.2,0,0.2,0.4,0.6])
    ax41.set_yticklabels([-0.2,0,0.2,0.4,0.6])
    ax43.yaxis.set_ticklabels([])
    ax43.yaxis.set_ticks_position('left')
    ax43.set_xlabel('Lag (s)')
    ax43.legend(bbox_to_anchor=(0.04, 1.6, 0.2, 0.2), loc='upper left')

    path1c='../Data/HD/ProcessedData/cellpair_example/'
    #为了和e和f对齐的wi=0.03
    m_all=np.load(path1c+'m_all_Wake.npy')
    vrange=0.5
    im=ax51.imshow(m_all,vmin=-vrange,vmax=vrange,cmap='bwr',aspect='auto')#cmap='seismic')#new_cmap)
    ax51.set_yticks(np.arange(0, m_all.shape[0]+1, 50))
    ax51.set_yticklabels(np.arange(0, m_all.shape[0]+1, 50)[::-1])
    ax51.set_ylabel('Cell pair #', labelpad=0.02)
    mid = m_all.shape[1] // 2
    ax51.set_xticks([0, mid, m_all.shape[1]-1])
    ax51.set_xticklabels([-5, 0, 5])
    ax51.set_xlabel('Lag (s)', labelpad=0.08)
    # r'$\it{GC-1}$
    ax51.set_title(r'$\it{HD}$ RUN')
    ax_pos = ax51.get_position()  
    cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0, 0.005, 0.5*ax_pos.height]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5,pad=0.0) 
    cbar.set_ticks([-vrange, -vrange/2, 0, vrange/2, vrange])
    cbar.set_ticklabels([' $\\mathrm{-}$'+'{:.1f}'.format(vrange), ' ', '    0.0', '', '    {:.1f}'.format(vrange)])
    

    m_all=np.load(path1c+'m_all_REM.npy')
    im=ax52.imshow(m_all,vmin=-vrange,vmax=vrange,cmap='bwr',aspect='auto')#cmap='seismic')#new_cmap)
    ax52.set_yticks(np.arange(0, m_all.shape[0]+1, 50))
    ax52.set_yticklabels(np.arange(0, m_all.shape[0]+1, 50)[::-1])
    # ax52.set_ylabel('Cell pair #', labelpad=0.02)
    mid = m_all.shape[1] // 2  
    ax52.set_xticks([0, mid, m_all.shape[1]-1])  
    ax52.set_xticklabels([-5, 0, 5]) 
    ax52.set_xlabel('Lag (s)', labelpad=0.08)
    ax52.set_title(r'$\it{HD}$ REM')
    
    ax_pos = ax52.get_position()  
    cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0, 0.005, 0.5*ax_pos.height]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5,pad=0.0) 
    cbar.set_ticks([-vrange, -vrange/2, 0, vrange/2, vrange])
    cbar.set_ticklabels([' $\\mathrm{-}$'+'{:.1f}'.format(vrange), ' ', '    0.0', '', '    {:.1f}'.format(vrange)])
    
    m_all=np.load(path1c+'m_all_SWS.npy')
    im=ax53.imshow(m_all,vmin=-vrange,vmax=vrange,cmap='bwr',aspect='auto')#cmap='seismic')#new_cmap)
    ax53.set_yticks(np.arange(0, m_all.shape[0]+1, 50))
    ax53.set_yticklabels(np.arange(0, m_all.shape[0]+1, 50)[::-1])
    # ax53.set_ylabel('Cell pair #', labelpad=0.02)
    mid = m_all.shape[1] // 2  
    ax53.set_xticks([0, mid, m_all.shape[1]-1])  
    ax53.set_xticklabels([-5, 0, 5]) 
    ax53.set_xlabel('Lag (s)', labelpad=0.08)
    ax53.set_title(r'$\it{HD}$ SWS')

    ax_pos = ax53.get_position()  
    cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0, 0.005, 0.5*ax_pos.height]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5,pad=0.0) 
    cbar.set_ticks([-vrange, -vrange/2, 0, vrange/2, vrange])
    cbar.set_ticklabels([' $\\mathrm{-}$'+'{:.1f}'.format(vrange), ' ', '    0.0', '', '    {:.1f}'.format(vrange)])
    
    path1d='../Data/GC1/cellpair_example/'
    s='1f20835f09e28706'
    m_all=np.load(path1d+s+'m_all.npy')
    vrange=0.6
    im=ax54.imshow(m_all,vmin=-vrange,vmax=vrange,cmap='bwr',aspect='auto')#cmap='seismic')#new_cmap)
    ax54.set_yticks(np.arange(0, m_all.shape[0]+1, 100))
    ax54.set_yticklabels(np.arange(0, m_all.shape[0]+1, 100)[::-1])
    ax54.set_ylabel('Cell pair #', labelpad=1)
    mid = m_all.shape[1] // 2
    ax54.set_xticks([0, mid, m_all.shape[1]-1])
    ax54.set_xticklabels([-10, 0, 10])
    ax54.set_xlabel('Lag (s)', labelpad=0.08)
    ax54.set_title(r'$\it{GC}$-1 OF')
    ax_pos = ax54.get_position()
    cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0, 0.005, 0.5*ax_pos.height]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5,pad=0.0)
    cbar.set_ticks([-vrange, -vrange/2, 0, vrange/2, vrange])
    cbar.set_ticklabels([' $\\mathrm{-}$'+'{:.1f}'.format(vrange), ' ', '    0.0', '', '    {:.1f}'.format(vrange)])
    
    path1d2 = '../Data/GC2/cellpair_example/'
    m='Kerala'
    s='1207_1'
    m_all=np.load(path1d2+m+s+'m_all.npy')
    im=ax55.imshow(m_all,vmin=-0.2,vmax=0.60,cmap='bwr',aspect='auto')#cmap='seismic')#new_cmap)
    ax55.set_yticks(np.arange(0, m_all.shape[0]+1, 100))
    ax55.set_yticklabels(np.arange(0, m_all.shape[0]+1, 100)[::-1])
    # ax55.set_ylabel('Cell pair #', labelpad=0.02)
    mid = m_all.shape[1] // 2
    ax55.set_xticks([0, mid, m_all.shape[1]-1])
    ax55.set_xticklabels([-10, 0, 10])
    ax55.set_xlabel('Lag (s)', labelpad=0.08)
    ax55.set_title(r'$\it{GC}$-2 1D Track')
    ax_pos = ax55.get_position()
    cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0, 0.005, 0.5*ax_pos.height]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5,pad=0.0) 
    cbar.set_ticks([-0.2, 0, 0.2,0.4,0.6])
    cbar.set_ticklabels([' $\\mathrm{-}$'+'0.2', '    0.0', '', '','    0.6'])
    
    path1e='../Data/HD/ProcessedData/RsDistribution/'
    # Code_readme/HD/RsDistributionAll.py 
    with open(path1e+'output'+'Wake'+'.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    r_all=np.array(output['r_all'])
    labels_all=output['labels_all']
    idx_hh=[n for n, x in enumerate(labels_all) if x == '$R_{hh}$']
    idx_hn=[n for n, x in enumerate(labels_all) if x == '$R_{hn}$']
    idx_nn=[n for n, x in enumerate(labels_all) if x == '$R_{nn}$']
    r_hh=r_all[idx_hh]
    r_hn=r_all[idx_hn]
    r_nn=r_all[idx_nn]
    vmax=0.4
    vmin=-0.4
    sns.kdeplot(r_hh, c='b', label='HH', ax=ax61,zorder=3,alpha=0.7)
    sns.kdeplot(r_hn, c='tab:orange', label='HN', ax=ax61,zorder=2,alpha=0.6)
    sns.kdeplot(r_nn, c='tab:gray', label='NN', ax=ax61,zorder=1,alpha=0.6)
    ax61.legend(facecolor='none',edgecolor='none')
    ax61.set_xlim(vmin,vmax)
    ax61.set_xlabel('Zero-lag correlation coefficient')
    ax61.set_ylim(0,20)
    ax61.set_ylabel('PDF')
    yticks = ax61.get_yticks()
    yticks = [0,5,10,15,20]
    ax61.set_yticks(yticks)
    ax61.set_title(r'$\it{HD}$ RUN')

    with open(path1e+'output'+'REM'+'.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    r_all=np.array(output['r_all'])
    labels_all=output['labels_all']
    idx_hh=[n for n, x in enumerate(labels_all) if x == '$R_{hh}$']
    idx_hn=[n for n, x in enumerate(labels_all) if x == '$R_{hn}$']
    idx_nn=[n for n, x in enumerate(labels_all) if x == '$R_{nn}$']
    r_hh=r_all[idx_hh]
    r_hn=r_all[idx_hn]
    r_nn=r_all[idx_nn]
    vmax=0.4
    vmin=-0.4
    sns.kdeplot(r_hh, c='b', label='HH', ax=ax62,zorder=3,alpha=0.7)#colors_tab10[0]
    sns.kdeplot(r_hn, c='tab:orange', label='HN', ax=ax62,zorder=2,alpha=0.6)#colors_tab10[3]
    sns.kdeplot(r_nn, c='tab:gray', label='NN', ax=ax62,zorder=1,alpha=0.6)#colors_tab10[7]
    ax62.legend(facecolor='none',edgecolor='none')
    ax62.set_xlim(vmin,vmax)
    ax62.set_ylim(0,20)
    ax62.set_xlabel('Zero-lag correlation coefficient')
    ax62.set_ylabel('')
    ax62.set_yticks(yticks)
    ax62.set_title(r'$\it{HD}$ REM')

    with open(path1e+'output'+'SWS'+'.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    r_all=np.array(output['r_all'])
    labels_all=output['labels_all']
    idx_hh=[n for n, x in enumerate(labels_all) if x == '$R_{hh}$']
    idx_hn=[n for n, x in enumerate(labels_all) if x == '$R_{hn}$']
    idx_nn=[n for n, x in enumerate(labels_all) if x == '$R_{nn}$']
    r_hh=r_all[idx_hh]
    r_hn=r_all[idx_hn]
    r_nn=r_all[idx_nn]
    vmax=0.4
    vmin=-0.4
    sns.kdeplot(r_hh, c='b', label='HH', ax=ax63,zorder=3,alpha=0.7)#colors_tab10[0]
    sns.kdeplot(r_hn, c='tab:orange', label='HN', ax=ax63,zorder=2,alpha=0.6)#colors_tab10[3]
    sns.kdeplot(r_nn, c='tab:gray', label='NN', ax=ax63,zorder=1,alpha=0.6)#colors_tab10[7]
    ax63.legend(facecolor='none',edgecolor='none')
    ax63.set_xlim(vmin,vmax)
    ax63.set_ylim(0,20)
    ax63.set_xlabel('Zero-lag correlation coefficient')
    ax63.set_ylabel('')
    ax63.set_yticks(yticks)
    ax63.set_title(r'$\it{HD}$ SWS')
    
    path1f='../Data/GC1/RsDistribution/'
    with open(path1f+'output.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    r_all=np.array(output['r_all'])
    labels_all=output['labels_all']
    idx_hh=[n for n, x in enumerate(labels_all) if x == '$R_{gg}$']
    idx_hn=[n for n, x in enumerate(labels_all) if x == '$R_{gn}$']
    idx_nn=[n for n, x in enumerate(labels_all) if x == '$R_{nn}$']
    r_hh=r_all[idx_hh]
    r_hn=r_all[idx_hn]
    r_nn=r_all[idx_nn]
    r_hh = r_hh[~np.isnan(r_hh)]
    r_hn = r_hn[~np.isnan(r_hn)]
    r_nn = r_nn[~np.isnan(r_nn)]
    vmax=0.4
    vmin=-0.25
    
    sns.kdeplot(r_hh, c='b', label='GG', ax=ax64,zorder=3,alpha=0.7)#colors_tab10[0]
    sns.kdeplot(r_hn, c='tab:orange', label='GN', ax=ax64,zorder=2,alpha=0.6)#colors_tab10[3]
    sns.kdeplot(r_nn, c='tab:gray', label='NN', ax=ax64,zorder=1,alpha=0.6)#colors_tab10[7]
    ax64.legend(facecolor='none',edgecolor='none')

    ax64.set_xlim(vmin,vmax)
    ax64.set_xlabel('Zero-lag correlation coefficient')
    ax64.set_ylabel('PDF')
    ax64.set_ylim(0,16)
    ax64.set_yticks(yticks)
    ax64.set_title('nF\nGC1')
    ax64.set_title(r'$\it{GC}$-1 OF')

    path1g='../Data/GC2/RsDistribution/'
    with open(path1g+'output.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    r_all=np.array(output['r_all'])
    labels_all=output['labels_all']
    idx_hh=[n for n, x in enumerate(labels_all) if x == '$R_{gg}$']
    idx_hn=[n for n, x in enumerate(labels_all) if x == '$R_{gn}$']
    idx_nn=[n for n, x in enumerate(labels_all) if x == '$R_{nn}$']
    r_hh=r_all[idx_hh]
    r_hn=r_all[idx_hn]
    r_nn=r_all[idx_nn]
    r_hh = r_hh[~np.isnan(r_hh)]
    r_hn = r_hn[~np.isnan(r_hn)]
    r_nn = r_nn[~np.isnan(r_nn)]
    vmax=0.4
    vmin=-0.4
    
    sns.kdeplot(r_hh, c='b', label='GG', ax=ax65,zorder=3,alpha=0.7)#colors_tab10[0]
    sns.kdeplot(r_hn, c='tab:orange', label='GN', ax=ax65,zorder=2,alpha=0.6)#colors_tab10[3]
    sns.kdeplot(r_nn, c='tab:gray', label='NN', ax=ax65,zorder=1,alpha=0.6)#colors_tab10[7]
    ax65.legend(facecolor='none',edgecolor='none')
    ax65.set_xlim(vmin,vmax)
    ax65.set_ylim(0,16)
    ax65.set_xlabel('Zero-lag correlation coefficient')
    ax65.set_ylabel('')
    ax65.set_yticks(yticks)
    # ax65.set_title('1D Track\nGC2')
    ax65.set_title(r'$\it{GC}$-2 1D Track')

    cmap_oranges = plt.cm.get_cmap('Oranges')
    cmap_binary = plt.cm.get_cmap('binary')
    colors_oranges = cmap_oranges(np.linspace(0, 1, 256))
    colors_binary = cmap_binary(np.linspace(0, 1, 256))
    combined_colors = np.vstack((colors_binary[0], colors_oranges))
    new_cmap_oranges = colors.ListedColormap(combined_colors)

    path1h='../Data/HD/ProcessedData/Q_quality/'
    with open(path1h+'output.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    im1=np.array(output['im1'])
    im2=np.array(output['im2'])
    im3=np.array(output['im3'])
    for i in range(im1.shape[0]):
        im1[i,i]=np.nan
        im2[i,i]=np.nan
        im3[i,i]=np.nan
    im3=im3*(im3>0.7)

    # for n in range(len(im3)):
    #     im3[n,n]=1
    vrange=0.5
    im=ax71.imshow(im1[7:28,7:28],vmin=-vrange,vmax=vrange,cmap='bwr')
    rect = patches.Rectangle((-0.5, -0.5), 13, 13, linewidth=1, edgecolor='k', facecolor='none')
    ax71.add_patch(rect)
    ax71.text(-2,-1,'$\\downarrow$',color='tab:blue',fontsize=10)
    ax71.text(0,-1,'$\\downarrow$',color='tab:cyan',fontsize=10)
    ax71.text(2,-1,'$\\downarrow$',color='tab:green',fontsize=10)
    ax71.text(12,-1,'$\\downarrow$',color='tab:purple',fontsize=10)
    ax71.text(15,-1,'$\\downarrow$',color='tab:brown',fontsize=10)
    ax71.text(-4.5,1,'$\\rightarrow$',color='tab:blue',fontsize=10)
    ax71.text(-4.5,3,'$\\rightarrow$',color='tab:cyan',fontsize=10)
    ax71.text(-4.5,5,'$\\rightarrow$',color='tab:green',fontsize=10)
    ax71.text(-4.5,15,'$\\rightarrow$',color='tab:purple',fontsize=10)
    ax71.text(-4.5,18,'$\\rightarrow$',color='tab:brown',fontsize=10)
    ax71.set_xticks([])
    ax71.set_yticks([])
    ax_pos = ax71.get_position()
    cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0, 0.005, 0.5*ax_pos.height]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5,pad=0.0)
    cbar.set_ticks([-vrange, 0, vrange])
    cbar.set_ticklabels([' $\\mathrm{-}$'+'{:.1f}'.format(vrange), '    0', '    {:.1f}'.format(vrange)])

    im=ax72.imshow(im2[7:28,7:28], cmap=new_cmap_oranges)
    rect = patches.Rectangle((-0.5, -0.5), 13, 13, linewidth=1, edgecolor='k', facecolor='none')
    ax72.add_patch(rect)
    ax72.text(-2,-1,'$\\downarrow$',color='tab:blue',fontsize=10)
    ax72.text(0,-1,'$\\downarrow$',color='tab:cyan',fontsize=10)
    ax72.text(2,-1,'$\\downarrow$',color='tab:green',fontsize=10)
    ax72.text(12,-1,'$\\downarrow$',color='tab:purple',fontsize=10)
    ax72.text(15,-1,'$\\downarrow$',color='tab:brown',fontsize=10)
    ax72.text(-4.5,1,'$\\rightarrow$',color='tab:blue',fontsize=10)
    ax72.text(-4.5,3,'$\\rightarrow$',color='tab:cyan',fontsize=10)
    ax72.text(-4.5,5,'$\\rightarrow$',color='tab:green',fontsize=10)
    ax72.text(-4.5,15,'$\\rightarrow$',color='tab:purple',fontsize=10)
    ax72.text(-4.5,18,'$\\rightarrow$',color='tab:brown',fontsize=10)
    ax72.set_xticks([])
    ax72.set_xlabel('Matrix $A$')
    ax72.set_yticks([])
    ax_pos = ax72.get_position()
    cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0, 0.005, 0.5*ax_pos.height]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_ticks([0,0.5,1])
    # cbar.set_ticklabels(['-{:.1f}'.format(vrange), '0', '{:.1f}'.format(vrange)])
    
    im=ax74.imshow(im3[7:28,7:28], cmap=plt.cm.gray_r)
    ax74.set_xticks([])
    ax74.set_xlabel('Matrix $B$')
    ax74.set_yticks([])
    ax74_pos = ax74.get_position() 
    # fig.text(ax_pos.x0+1/8*ax_pos.width, ax_pos.y1-1/4*ax_pos.width, 'C*C', color='white', fontsize=12)
    # cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0+ 0.003, 0.005, 0.5*ax_pos.height]) 
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.ax.tick_params(labelsize=5) 
    # cbar.set_ticks([0,0.5,1])
    
    
    # 创建一个新的空图
    G = nx.Graph()

    # 添加核心节点，编号从1到13
    core_nodes = range(1, 13)
    G.add_nodes_from(core_nodes)

    # 添加边缘节点，编号从14到21
    periphery_nodes = range(14, 21)
    G.add_nodes_from(periphery_nodes)

    # 连接核心节点，并给这些边一个较小的权重
    for i in range(1, 21):
        for j in range(i, 21):
            if i < j:  # 避免自环和重复添加边
                G.add_edge(i, j, weight=im2[6+i,6+j])  # 较小的权重使得核心节点之间不太紧密

    # # 将边缘节点连接到随机选择的核心节点，并给这些边一个较大的权重
    # for node in periphery_nodes:
    #     G.add_edge(node, random.choice(core_nodes), weight=3.5)  # 较大的权重使得边缘节点更接近核心

    # 使用spring布局，调整节点间的引力和斥力
    pos = nx.spring_layout(G, k=1.3, iterations=50, weight='weight')  # k参数可以微调节点间的距离

    # 绘制核心节点
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=core_nodes, 
        node_size=20, 
        node_color=['tab:blue','gray','tab:cyan','gray','tab:green','gray','gray','gray','gray','gray','gray','gray'], 
        edgecolors=['tab:blue','gray','tab:cyan','gray','tab:green','gray','gray','gray','gray','gray','gray','gray'],  # 设置节点的边缘颜色为灰色
        label='Core Nodes',
        ax=ax73
    )

    # 绘制边缘节点
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=periphery_nodes, 
        node_size=20, 
        node_color='none', 
        edgecolors=['gray','tab:purple','gray','gray','tab:brown','gray','gray',],  # 设置节点的边缘颜色为灰色
        label='Periphery Nodes',
        ax=ax73
    )

    # 选择性地绘制权重大于某个阈值的边
    threshold = 0.2  # 设置阈值
    edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
    edges_to_draw_weight=[d['weight'] for u, v, d in G.edges(data=True) if d['weight'] > threshold]
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw,edge_color=edges_to_draw_weight,edge_cmap=new_cmap_oranges,edge_vmin=0,edge_vmax=1, alpha=1, ax=ax73)
    ax73.axis('off')  # 关闭坐标轴


    path1i1='../Data/HD/ProcessedData/NullModel_new/'
    path1i2='../Data/GC1/NullModel_new/'
    path1i3='../Data/GC2/NullModel_new/'
    with open(path1i1+'output.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    Q_s_values=output['Q_s_values']
    Q=output['Q']
    ax81.hist(Q_s_values, bins=np.linspace(20,75,31), alpha=1, color='grey', label='Shuffled Q')
    ax81.axvline(Q, color='blue', linestyle='dashed', linewidth=1.5, label='nriginal Q')  # 画出原始Q值的位置
    ax81.set_xlabel(r'$R’$',labelpad=0.1)
    ax81.set_xlim(20,75)
    ax81.set_ylabel('Counts', labelpad=0.04)
    # yticks = [0,5,10]
    # ax81.set_yticks(yticks)
    ax81.set_title(r'$\it{HD}$ RUN')

    with open(path1i2+'output.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    Q_s_values=output['Q_s_values']
    Q=output['Q']
    ax82.hist(Q_s_values, bins=np.linspace(26,51,31),color='grey',label='Shuffled Q')
    ax82.axvline(Q, color='blue', linestyle='dashed', linewidth=1.5, label='nriginal Q')  # 画出原始Q值的位置
    ax82.set_xlabel(r'$R’$',labelpad=0.1)
    ax82.set_xlim(26,51)
    # yticks = [0,5,10,15]
    # ax82.set_yticks(yticks)
    ax82.set_title(r'$\it{GC}$-1 OF')

    with open(path1i3+'output.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    Q_s_values=output['Q_s_values']
    Q=output['Q']
    ax83.hist(Q_s_values, bins=np.linspace(140,290,31),color='grey',label='Shuffled Q')
    ax83.axvline(Q, color='blue', linestyle='dashed', linewidth=1.5, label='nriginal Q')  # 画出原始Q值的位置
    ax83.set_xlabel(r'$R’$',labelpad=0.1)
    ax83.set_xlim(140,290)
    # yticks = [0,5,10]
    # ax83.set_yticks(yticks)
    ax83.set_title(r'$\it{GC}$-2 1D Track')

    plt.figtext(ax_pos11.x0-0.035,ax_pos11.y1+0.02,'A',fontsize=15)
    plt.figtext(ax_pos31.x0-0.03,ax_pos31.y1,'B',fontsize=15)
    plt.figtext(ax_pos11.x0-0.035,ax_pos51.y1+0.01,'C',fontsize=15)
    plt.figtext(ax_pos54.x0-0.055,ax_pos54.y1,'D',fontsize=15)
    plt.figtext(ax_pos11.x0-0.035,ax_pos61.y1+0.01,'E',fontsize=15)
    plt.figtext(ax_pos64.x0-0.03,ax_pos64.y1+0.01,'F',fontsize=15)
    plt.figtext(ax_pos11.x0-0.035,ax_pos71.y1+0.003,'G',fontsize=15)
    plt.figtext(ax_pos11.x0+0.12,ax_pos71.y1+0.003,'H',fontsize=15)
    plt.figtext(ax_pos11.x0+0.28,ax_pos71.y1+0.003,'I',fontsize=15)
    plt.figtext(ax74_pos.x0-0.01,ax_pos71.y1+0.003,'J',fontsize=15)
    plt.figtext(ax_pos81.x0-0.03,ax_pos71.y1+0.003,'K',fontsize=15)

    plt.savefig('../Figures/Figure1.png',format='PNG',dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()