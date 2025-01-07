#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:00:50 2024

@author: qianruixin
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
from statannotations.Annotator import Annotator
from scipy import stats

import random
import networkx as nx
import sys
sys.path.append('../') # 将上级目录加入 sys.path
from DrawStandard import *
import pickle
from DetectCP import GenerateGraph, CalCoreness, Metrics, DrawNetwork
import seaborn as sns

from sklearn.linear_model import RANSACRegressor

def ransac_filter(x, y, threshold=1):
    # RANSACRegressor类的阈值参数默认设置为3.0
    # Reshape data
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    # Run RANSAC algorithm
    ransac = RANSACRegressor()
    ransac.fit(x, y)
    residual_errors = np.abs(ransac.predict(x) - y)

    # Get inlier mask based on threshold
    inlier_mask = residual_errors < threshold

    # Get inlier data
    x_inlier = x[inlier_mask]
    y_inlier = y[inlier_mask]

    # Get slope and intercept
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    return x_inlier, y_inlier, slope, intercept


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
    # path='../Data/GC1/Fig3_supp_csv/'
    path='../Data/GC-1-supp/GS_CS_MI_'
    session_list=['8a50a33f7fd91df4','1f20835f09e28706','0de4b55d27c9f60f','8f7ddffaf4a5f4c5', '7e888f1d8eaab46b','5b92b96313c3fc19','59825ec5641c94b4','c221438d58a0b796']
    FigIDs=['A','B','C','D','E','F','G','H']
    for no,s in enumerate(session_list):
        if no%4==0:
            plt.figtext(0.05,0.76-0.24*int(no/4)+0.17,r'$\it{GC}$-1 OF '+s)
            ax5a=fig.add_axes([0.05,0.76-0.24*int(no/4),0.08,0.16])
            ax5b=fig.add_axes([0.14,0.76-0.24*int(no/4),0.08,0.16])
        if no%4==1:
            plt.figtext(0.29,0.76-0.24*int(no/4)+0.17,r'$\it{GC}$-1 OF '+s)
            ax5a=fig.add_axes([0.29,0.76-0.24*int(no/4),0.08,0.16])
            ax5b=fig.add_axes([0.38,0.76-0.24*int(no/4),0.08,0.16])
        if no%4==2:
            plt.figtext(0.55,0.76-0.24*int(no/4)+0.17,r'$\it{GC}$-1 OF '+s)
            ax5a=fig.add_axes([0.55,0.76-0.24*int(no/4),0.08,0.16])
            ax5b=fig.add_axes([0.64,0.76-0.24*int(no/4),0.08,0.16])
        if no%4==3:
            plt.figtext(0.79,0.76-0.24*int(no/4)+0.17,r'$\it{GC}$-1 OF '+s)
            ax5a=fig.add_axes([0.79,0.76-0.24*int(no/4),0.08,0.16])
            ax5b=fig.add_axes([0.88,0.76-0.24*int(no/4),0.08,0.16])
        # s='8a50a33f7fd91df4'
        df=pd.read_csv(path+s+'.csv',index_col=0)
        print(df)
        
        color_dict = {'NGC': 'tab:olive', 'GC': 'tab:cyan'}
        sns.stripplot(data=df, x="GCG", y="r_1st_2nd_RM", order=['GC', 'NGC'], palette=color_dict, ax=ax5a, alpha=0.5)
        sns.boxplot(data=df, x="GCG", y="r_1st_2nd_RM", order=['GC', 'NGC'], palette=color_dict, ax=ax5a, boxprops={'facecolor':'None'}, showfliers=False)
        pairs = [("GC", "NGC")]
        annotator = Annotator(ax5a, pairs, data=df, x="GCG", y="r_1st_2nd_RM", order=['GC', 'NGC'],)
        annotator.configure(test='t-test_ind', text_format='star', line_height=0.03, line_width=1)
        annotator.apply_and_annotate()
        
        color_dict = {'Core': 'tab:red', 'Peri': 'tab:blue'}
        sns.stripplot(data=df, x="CPG", y="r_1st_2nd_RM", order=['Core', 'Peri'], palette=color_dict, ax=ax5b, alpha=0.5)
        sns.boxplot(data=df, x="CPG", y="r_1st_2nd_RM", order=['Core', 'Peri'], palette=color_dict, ax=ax5b, boxprops={'facecolor':'None'}, showfliers=False)
        pairs = [("Core", "Peri")]
        annotator = Annotator(ax5b, pairs, data=df, x="CPG", y="r_1st_2nd_RM", order=['Core', 'Peri'],)
        annotator.configure(test='t-test_ind', text_format='star', line_height=0.03, line_width=1)
        annotator.apply_and_annotate()
        
        ax5a.set_xticks([0,1])
        ax5a.set_xticklabels(['GCs','nGCs'])
        ax5a.set_xlabel('')
        ax5a.set_ylabel('Correlation between the\n1st- and 2nd-half rate map',labelpad=0.1)
        ax5a.spines["top"].set_visible(False)
        ax5a.spines["right"].set_visible(False)
        
        ax5b.set_xticks([0,1])
        ax5b.set_xticklabels(['Core','Periphery'])
        ax5b.set_xlabel('')
        ax5b.set_yticklabels([])
        ax5b.set_ylabel('')
        ax5b.spines["top"].set_visible(False)
        ax5b.spines["right"].set_visible(False)


        ax5a_pos=ax5a.get_position()
        plt.figtext(ax5a_pos.x0-0.025,ax5a_pos.y1+0.008,FigIDs[no],fontsize=15)

    
    plt.savefig('../Figures/SuppFigure11.png',format='PNG',dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()