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
    IDs = ['Mouse12-120806', 'Mouse12-120807', 'Mouse12-120808', 'Mouse12-120809', 'Mouse12-120810',
      'Mouse20-130517', 'Mouse28-140313', 'Mouse25-140130']
    # Clus = ['1_8', '1_8', '1_8', '1_8', '1_8', '1_8', '8_11', '5_8']
    States = ['Wake', 'REM', 'SWS']
    StateNames = ['RUN', 'REM', 'SWS']
    pathHD='../Data/HD/ProcessedData/NullModel_new/'
    info_all=np.load(pathHD+'info_all.npy',allow_pickle=True).item()
    count=0
    num_rows=9
    num_columns=4
    panel_width=0.2
    panel_height=0.08
    panel_wgap=0.03
    panel_hgap=0.02
    for i in range(len(IDs)):
        for j in range(len(States)):
            FilePath = IDs[i]
            State = States[j]
            Q_s_values = info_all[FilePath+'_'+State]['Q_s_values']
            Q = info_all[FilePath+'_'+State]['Q']
            count += 1
            # ax = fig.add_axes([num_rows, num_columns, count])
            xid = (count-1)%4
            yid = 8-int((count-1)/4)
            ax = fig.add_axes([0.05+panel_width*xid+panel_wgap*xid,0.05+panel_height*yid+panel_hgap*yid,panel_width,panel_height])
            plt.hist(Q_s_values, bins=20, color='grey', label='Shuffled Q')
            plt.axvline(Q, color='blue', linestyle='dashed', linewidth=2, label='Original Q')  # 画出原始Q值的位置
            # print(ax.get_xlim()[0],ax.get_xlim()[1])
            ax.text(ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.9,ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.9,IDs[i]+' '+StateNames[j],ha='right',va='top')
            # plt.legend()

            # Check if the subplot is in the last row
            if count > (num_rows - 1) * num_columns:
                ax.set_xlabel(r'$R$',labelpad=0.1)
            
            # Check if the subplot is in the first column
            if count % num_columns == 1:
                ax.set_ylabel('Counts',labelpad=0.1)

            # # Optionally, remove x-tick labels for all but the last row of subplots
            # if count <= (num_rows - 1) * num_columns:
            #     plt.setp(ax.get_xticklabels(), visible=False)
            
            # # Optionally, remove y-tick labels for all but the first column of subplots
            # if count % num_columns != 1:
            #     plt.setp(ax.get_yticklabels(), visible=False)

    session_list=['8a50a33f7fd91df4','1f20835f09e28706','0de4b55d27c9f60f','8f7ddffaf4a5f4c5', '7e888f1d8eaab46b','5b92b96313c3fc19','59825ec5641c94b4','c221438d58a0b796']
    pathGC1='../Data/GC1/NullModel_new/'
    info_all=np.load(pathGC1+'info_all.npy',allow_pickle=True).item()
    # plt.figure(figsize=(8,4))
    
    for s in session_list:
        Q_s_values = info_all[s]['Q_s_values']
        Q = info_all[s]['Q']
        count += 1
        xid = (count-1)%4
        yid = 8-int((count-1)/4)
        ax = fig.add_axes([0.05+panel_width*xid+panel_wgap*xid,0.05+panel_height*yid+panel_hgap*yid,panel_width,panel_height])
        plt.hist(Q_s_values, bins=20, color='darkgrey', label='Shuffled Q')
        plt.axvline(Q, color='blue', linestyle='dashed', linewidth=2, label='Original Q')  # 画出原始Q值的位置
        # plt.legend()
        # print(ax.get_xlim()[0],ax.get_xlim()[1])
        ax.text(ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.9,ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.9,r'$\it{GC}$-1 OF'+'\n'+s,ha='right',va='top')
        
        # Check if the subplot is in the last row
        if count > (num_rows - 1) * num_columns:
            ax.set_xlabel(r'$R$',labelpad=0.1)
        
        # Check if the subplot is in the first column
        if count % num_columns == 1:
            ax.set_ylabel('Counts',labelpad=0.1)

        # # Optionally, remove x-tick labels for all but the last row of subplots
        # if count <= (num_rows - 1) * num_columns:
        #     plt.setp(ax.get_xticklabels(), visible=False)
        
        # # Optionally, remove y-tick labels for all but the first column of subplots
        # if count % num_columns != 1:
        #     plt.setp(ax.get_yticklabels(), visible=False)

    pathGC2='../Data/GC2/NullModel_new/'
    session_list= ['Mumbai_1201_1','Kerala_1207_1','Goa_1210_1','Punjab_1217_1']
    info_all=np.load(pathGC2+'info_all.npy',allow_pickle=True).item()
    # plt.figure(figsize=(8,2))
    for s in session_list:
        Q_s_values = info_all[s]['Q_s_values']
        Q = info_all[s]['Q']
        count += 1
        xid = (count-1)%4
        yid = 8-int((count-1)/4)
        ax = fig.add_axes([0.05+panel_width*xid+panel_wgap*xid,0.05+panel_height*yid+panel_hgap*yid,panel_width,panel_height])
        plt.hist(Q_s_values, bins=20, color='lightgrey', label='Shuffled Q')
        plt.axvline(Q, color='blue', linestyle='dashed', linewidth=2, label='Original Q')  # 画出原始Q值的位置
        # plt.legend()
        # print(ax.get_xlim()[0],ax.get_xlim()[1])
        ax.text(ax.get_xlim()[0]+(ax.get_xlim()[1]-ax.get_xlim()[0])*0.9,ax.get_ylim()[0]+(ax.get_ylim()[1]-ax.get_ylim()[0])*0.9,r'$\it{GC}$-2'+'\n'+s,ha='right',va='top')

        # Check if the subplot is in the last row
        if count > (num_rows - 1) * num_columns:
            ax.set_xlabel(r'$R’$',labelpad=0.1)
        
        # Check if the subplot is in the first column
        if count % num_columns == 1:
            ax.set_ylabel('Counts',labelpad=0.1)

        # # Optionally, remove x-tick labels for all but the last row of subplots
        # if count <= (num_rows - 1) * num_columns:
        #     plt.setp(ax.get_xticklabels(), visible=False)
        
        # # Optionally, remove y-tick labels for all but the first column of subplots
        # if count % num_columns != 1:
        #     plt.setp(ax.get_yticklabels(), visible=False)

    plt.savefig('../Figures/SuppFigure2.png',format='PNG',dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()