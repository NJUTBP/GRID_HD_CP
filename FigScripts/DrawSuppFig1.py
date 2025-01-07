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
    # Load the data from 'rall.npy'
    r_all = np.load('../Data/r_multiRegions.npy', allow_pickle=True).item()

    # Extract the data for the plot
    r_all_hd = r_all['data'][0]
    r_all_gc1 = r_all['data'][1]
    r_all_gc2 = r_all['data'][2]
    r_all_gc3 = r_all['data'][3]
    r_all_hc = r_all['data'][4]
    r_all_pfc = r_all['data'][5]
    labels_all = r_all['labels']

    # Define the colors for the boxes
    box_colors = ['lightskyblue', 'orange', 'lightgreen', 'lightpink', 'plum', 'tan']

    # Create the violin plot
    
    fig=plt.figure(figsize=(8.5,11))
    ax = fig.add_axes([0.05,0.75,0.40,0.20])
    data = [r_all_hd, r_all_gc1, r_all_gc2, r_all_gc3, r_all_hc, r_all_pfc]
    vp = sns.violinplot(data=data, palette=box_colors, inner=None,scale='width', cut=0)
    ax.hlines(0,-0.5,6.5,linestyle='--',color='gray')
    # Set custom tick labels with italics
    ax.set_xticklabels(
        [r'$\it{HD}$', r'$\it{GC}$-1', r'$\it{GC}$-2', r'$\it{GC}$-3', r'$\it{hc}$-17', r'$\it{pfc}$-8'],
    )
    # Set the y-axis label
    ax.set_ylabel('Correlation coefficient',labelpad=0.1)
    ax.set_xlim(-0.5,5.5)

    # Tight layout for better spacing
    # plt.tight_layout()
    
    plt.savefig('../Figures/SuppFigure1.png',format='PNG',dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()