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
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import minimize
def gaussian(x, a, b, c, d):
    return  a * np.exp(-(x - b)**2 / (2 * c**2)) - d

def fit_gaussian(x, y):
    # Provide an initial guess for parameters a, b, c, d
    initial_guess = [1, 0, 1, 1]
    
    # Perform curve fit
    popt, pcov = curve_fit(gaussian, x, y, p0=initial_guess)

    # popt contains the optimal values for a, b, c, and d
    a, b, c, d = popt
    
    # Return the optimized parameters
    return a, b, c, d

def fit_and_plot_gaussian(xdata, ydata):
    # 在x轴上翻转数据
    xdata_flipped = 2 * np.min(xdata) - xdata
    ydata_flipped = ydata
    # 合并原始数据和翻转后的数据
    xdata_combined = np.concatenate((xdata, xdata_flipped))
    ydata_combined = np.concatenate((ydata, ydata_flipped))

    # 对合并后的数据进行排序
    sort_indices = np.argsort(xdata_combined)
    xdata_combined = xdata_combined[sort_indices]
    ydata_combined = ydata_combined[sort_indices]

    # 画出原始数据

    a, b, c, d = fit_gaussian(xdata_combined, ydata_combined)
    
    x=np.linspace(min(xdata_combined),max(xdata_combined),1000)
    x=x[int(0.5*len(x)):]
    y = gaussian(x, a, b, c, d)
    # plt.figure()
    # plt.plot(xdata_combined, ydata_combined, 'b+', label='data')
    # plt.plot(x, y, 'r*', label='fit')
    
    # plt.legend()
    # plt.show()
    # y[y>1]=1
    return x,y

def fit_and_plot_gaussian2(xdata, ydata):
    # 在x轴上翻转数据
    xdata_flipped = 2 * np.min(xdata) - xdata
    ydata_flipped = ydata
    # 合并原始数据和翻转后的数据
    xdata_combined = np.concatenate((xdata, xdata_flipped))
    ydata_combined = np.concatenate((ydata, ydata_flipped))

    # 对合并后的数据进行排序
    sort_indices = np.argsort(xdata_combined)
    xdata_combined = xdata_combined[sort_indices]
    ydata_combined = ydata_combined[sort_indices]

    # 画出原始数据

    a, b, c, d = fit_gaussian(xdata_combined, ydata_combined)
    
    x=np.linspace(min(xdata_combined),max(xdata_combined),1000)
    x=x[int(0.5*len(x)):]
    y = gaussian(x, a, b, c, d)
    # plt.figure()
    # plt.plot(xdata_combined, ydata_combined, 'b+', label='data')
    # plt.plot(x, y, 'r*', label='fit')
    
    # plt.legend()
    # plt.show()
    y/=np.max(y)
    return x,y

def main():
    fig=plt.figure(figsize=(8.5,11))
    ax1=fig.add_axes([0.050,0.85,0.20,0.10])
    ax2=fig.add_axes([0.05,0.725,0.20,0.10])
    ax3=fig.add_axes([0.05,0.60,0.20,0.10])

    ax4=fig.add_axes([0.30,0.85,0.20,0.10])
    ax5=fig.add_axes([0.30,0.70,0.20,0.10])
    ax6=fig.add_axes([0.30,0.60,0.20,0.10])

    ax7=fig.add_axes([0.54,0.85,0.20,0.10])
    ax8=fig.add_axes([0.54,0.70,0.20,0.10])
    ax9=fig.add_axes([0.54,0.60,0.20,0.10])
    
    axa=fig.add_axes([0.05,0.45,0.10*11/8.5,0.10])
    axb=fig.add_axes([0.22,0.45,0.10*11/8.5,0.10])
    axc=fig.add_axes([0.41,0.45,0.12,0.1])
    axd=fig.add_axes([0.56,0.45,0.18,0.1])

    path1='../Data/HD/ProcessedData/cellpair_spr/'
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
    # ax1.set_xlabel('Zero-lag correlation coefficient',labelpad=1)
    ax1.set_ylabel('$\\Delta \\theta$ (rad)',labelpad=0.1)
    # ax1.set_title(r'$\it{HD}$',pad=0.1)
    ax1.set_xlim(-0.35,0.95)
    ax1.set_xticks([-0.3,0,0.3,0.6,0.9])

    path2='../Data/GC1/cellpair_spr/'
    s='5b92b96313c3fc19'
    with open(path2+s+'.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    dists_s=output['dists_s']
    r0_s=output['r0_s']
    ax2.scatter(r0_s, dists_s, edgecolor=colors[0],facecolor='none', alpha=0.5,s=7)
    # ax2.set_xlabel('Zero-lag correlation coefficient',labelpad=1)
    ax2.set_ylabel('$|\\Delta \\bf{g}|$ (rad)',labelpad=0.1)
    # ax2.set_title(r'$\it{GC-1}$',pad=0.1)
    ax2.text(0.6,3.5,r'$\it{GC}$-1')
    ax2.set_xlim(-0.2,0.8)

    path3='../Data/GC3/cellpair_spr/'
    m='Q_2_'
    states=['OF','WW','REM','SWS']
    for no,s in enumerate(states):
        with open(path3+m+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        dists_s=output['dists_s']
        r0_s=output['r0_s']
        ax3.scatter(r0_s, np.array(dists_s)*np.pi, marker=marker[no],edgecolor=colors[no],facecolor='none', alpha=0.5,s=7,label=states[no])
    ax3.legend(facecolor='none',edgecolor='none',title=r'$\it{GC}$-3',labelspacing=0.5,handletextpad=0.1)
    ax3.set_xlabel('Correlation coefficient',labelpad=1)
    ax3.set_ylabel('$|\\Delta \\bf{g}|$ (rad)',labelpad=0.1)
    # ax3.set_title(r'$\it{GC-3}$',pad=0.1)
    ax3.set_xlim(-0.2,0.8)

    path4='../Data/HD/ProcessedData/predict_d/'    
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
    ax4.scatter(r2, d2, s=15, facecolor='red',edgecolor='none',alpha=0.005)
    x,y = fit_and_plot_gaussian(d2, r2)
    ax4.plot(y, x, c='k',linestyle='--')
    # ax41.plot(y, x, c='grey',linestyle='--')
    # ax41.plot(model_table[:,0], model_table[:,1], c=colors_tab10[6],linestyle='--')
    model_table={'r':y,'d':x}
    np.save(path4+'model_table_gaussian',model_table)
    ax4.set_xlabel('Correlation coefficient',labelpad=1)
    ax4.set_ylabel('$\\Delta \\theta$ (rad)',labelpad=0.1)
    ax4.set_title('HDCs in $SD$',pad=3.0)

    ax6.scatter(r1, d1, s=3,marker='o', facecolor='none',edgecolor='tab:blue',alpha=0.3)
    # ax6.scatter(r3, d3, s=15, facecolor='tab:orange',edgecolor='none',alpha=1)
    ax6.plot(y, x, c='tab:orange',linestyle='--',label='$SD$ curve')
    ax6.scatter(r1[11], d1[11], s=25,marker='*', facecolor='tab:red',edgecolor='tab:red',alpha=1,zorder=5)
    ax6.scatter(r3[11], d3[11], s=25,marker='o', facecolor='tab:green',edgecolor='tab:green',alpha=1,zorder=5)
    ax6.scatter(r1[19], d1[19], s=25,marker='*', facecolor='tab:red',edgecolor='tab:red',alpha=1,zorder=5)
    ax6.scatter(r3[19], d3[19], s=25,marker='o', facecolor='tab:green',edgecolor='tab:green',alpha=1,zorder=5)
    ax6.scatter(r1[11], d3[11], s=25,marker='^', facecolor='tab:green',edgecolor='tab:green',alpha=1,zorder=5)
    ax6.scatter(r1[19], d3[19], s=25,marker='^', facecolor='tab:green',edgecolor='tab:green',alpha=1,zorder=5)
    ax6.arrow(r3[11],d3[11],r1[11]-r3[11],0,facecolor='k',edgecolor='k',linewidth=1.0,length_includes_head=True,head_width=0.075*1.619,head_length=0.075,zorder=5)
    ax6.arrow(r3[19],d3[19],r1[19]-r3[19],0,facecolor='k',edgecolor='k',linewidth=1.0,length_includes_head=True,head_width=0.075*1.619,head_length=0.075,zorder=5)
    # ax6.annotate('', xy=(r1[11],d1[11]), xytext=(r3[11],d3[11]),  arrowprops=dict(facecolor='k',edgecolor='k',width=0.2,headwidth=3,headlength=6),zorder=5)
    # ax6.annotate('', xy=(r1[19],d1[19]), xytext=(r3[19],d3[19]),  arrowprops=dict(facecolor='k',edgecolor='k',width=0.2,headwidth=3,headlength=6),zorder=5)


    ax6.vlines(r3[11],-1,3.5,color='tab:green',linestyle=':')
    ax6.vlines(r3[19],-1,3.5,color='tab:green',linestyle=':')
    ax6.vlines(r1[11],-1,3.5,color='tab:red',linestyle=':')
    ax6.vlines(r1[19],-1,3.5,color='tab:red',linestyle=':')
    ax6.legend(loc='upper center',facecolor='none',edgecolor='none')
    ax6.set_xlim(-0.5,1.1)
    ax6.set_xticks([-0.5,0,0.5,1.0])
    ax6.set_xlabel('Correlation coefficient',labelpad=1)
    ax6.set_ylim(-0.1,3.3)
    ax6.set_ylabel('$\\Delta \\theta$ (rad)',labelpad=0.1)
    
    # sns.kdeplot(r1, fill=True, color='tab:purple', label='original',ax=ax5) # experiment
    # sns.kdeplot(r3, fill=True, color='tab:orange', label='aligned',ax=ax5)
    sns.ecdfplot(r3, color='k', linestyle='-',linewidth=2.5, label='$SD$ CDF',ax=ax5,alpha=0.7,zorder=-1)
    sns.ecdfplot(r1, color='tab:blue', label='original',ax=ax5) # experiment
    sns.ecdfplot(r3, color='tab:orange', linestyle='--', label='aligned',ax=ax5)
    # ax5.annotate('', xy=(-0.55, 0.5), xytext=(0.7, 0.5), 
    #             arrowprops=dict(arrowstyle="<->", color='black'))
    ax5.arrow(r1[11],0.28,r3[11]-r1[11]+0.02,0,facecolor='k',edgecolor='k',linewidth=1.0,length_includes_head=True,head_width=0.0375,head_length=0.075,zorder=5)
    ax5.arrow(r1[19],0.97,r3[19]-r1[19]-0.02,0,facecolor='k',edgecolor='k',linewidth=1.0,length_includes_head=True,head_width=0.0375,head_length=0.075,zorder=5)
    ax5.vlines(r1,0,0.1, color='tab:blue',linewidth=0.2,alpha=0.5)
    ax5.vlines(r3,0,0.1, color='tab:orange',linewidth=0.2,alpha=0.5)
    ax5.vlines(r1[11],0,0.1, color='tab:red',linewidth=1,alpha=1)
    ax5.vlines(r3[11],0,0.1, color='tab:green',linewidth=1,alpha=1)
    ax5.vlines(r1[19],0,0.1, color='tab:red',linewidth=1,alpha=1)
    ax5.vlines(r3[19],0,0.1, color='tab:green',linewidth=1,alpha=1)
    ax5.vlines(r1[11],0.1,0.28,linestyle=':', color='tab:red',alpha=1)
    ax5.vlines(r3[11],0.1,0.28,linestyle=':', color='tab:green',alpha=1)
    ax5.vlines(r1[19],0.1,0.97,linestyle=':', color='tab:red',alpha=1)
    ax5.vlines(r3[19],0.1,0.97,linestyle=':', color='tab:green',alpha=1)
    ax5.legend(loc='upper left',facecolor='none',edgecolor='none')
    ax5.set_xlim(-0.5,1.1)
    ax5.set_xticks([-0.5,0,0.5,1.0],[])
    # ax5.set_xlabel('Zero-lag correlation coefficient',labelpad=1)
    ax5.set_yticks([0,0.5,1])
    ax5.set_ylabel('CDF',labelpad=0.1)
    ax5.set_ylim(0,1.05)
    ax5.set_title('one session in $HD$',pad=3.0)

    axa.scatter(d1, d3, c='tab:blue', alpha=0.5,s=7)
    axa.plot([0,3.15], [0,3.15], c='grey', linewidth=2)
    axa.set_xlim([0, 3.15])
    axa.set_ylim([0, 3.15])
    axa.set_yticks(axa.get_xticks()[:-1])
    axa.set_xlabel('Actual $\\Delta \\theta$ (rad)',labelpad=0.1)
    axa.set_ylabel('Estimated $\\Delta \\theta$ (rad)',labelpad=0.1)  
    axa.set_title('one session from $HD$',pad=2.0)
    axa.text(0.35,2.8,'$r$ = {:.2f}'.format(pearsonr(d1, d3)[0]))


    path5='../Data/GC1/predict_d/'    
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
    ax7.scatter(r2, d2, s=15, facecolor='red',edgecolor='none',alpha=0.005)
    x,y = fit_and_plot_gaussian2(d2, r2)
    ax7.plot(y, x, c='k',linestyle='--')
    model_table={'r':y,'d':x}
    np.save(path5+'model_table_gaussian',model_table)

    ax7.set_xlim(-0.5,1.05)
    ax7.set_ylabel('$|\\Delta \\bf{g}|$ (rad)',labelpad=0.1)
    ax7.set_xlabel('Correlation coefficient',labelpad=1)
    ax7.set_title('GCs in $SD$',pad=3.0)
    
    print(len(r1),len(d1),len(r2),len(d2),len(r3),len(d3))
    ax9.scatter(r1_all, d1_all, s=3,marker='o', facecolor='none',edgecolor='tab:blue',alpha=0.3)
    # ax9.scatter(r3_all, d3_all, s=15, facecolor='tab:orange',edgecolor='none',alpha=1)
    ax9.plot(y, x, c='tab:orange',linestyle='--')
    ax9.scatter(r1_all[18], d1_all[18], s=25,marker='*', facecolor='tab:red',edgecolor='tab:red',alpha=1,zorder=5)
    ax9.scatter(r3_all[18], d3_all[18], s=25,marker='o', facecolor='tab:green',edgecolor='tab:green',alpha=1,zorder=5)
    ax9.scatter(r1_all[20], d1_all[20], s=25,marker='*', facecolor='tab:red',edgecolor='tab:red',alpha=1,zorder=5)
    ax9.scatter(r3_all[20], d3_all[20], s=25,marker='o', facecolor='tab:green',edgecolor='tab:green',alpha=1,zorder=5)
    ax9.scatter(r1_all[18], d3_all[18], s=25,marker='^', facecolor='tab:green',edgecolor='tab:green',alpha=1,zorder=5)
    ax9.scatter(r1_all[20], d3_all[20], s=25,marker='^', facecolor='tab:green',edgecolor='tab:green',alpha=1,zorder=5)
    ax9.arrow(r3_all[18],d3_all[18],r1_all[18]-r3_all[18],0,facecolor='k',edgecolor='k',linewidth=1.0,length_includes_head=True,head_width=0.20,head_length=0.07,zorder=5)
    ax9.arrow(r3_all[20],d3_all[20],r1_all[20]-r3_all[20],0,facecolor='k',edgecolor='k',linewidth=1.0,length_includes_head=True,head_width=0.20,head_length=0.07,zorder=5)
    # ax9.annotate('', xy=(r1_all[18],d1_all[18]), xytext=(r3_all[18],d3_all[18]),  arrowprops=dict(facecolor='k',edgecolor='k',width=0.2,headwidth=3,headlength=6),zorder=5)
    # ax9.annotate('', xy=(r1_all[20],d1_all[20]), xytext=(r3_all[20],d3_all[20]),  arrowprops=dict(facecolor='k',edgecolor='k',width=0.2,headwidth=3,headlength=6),zorder=5)

    ax9.vlines(r3_all[18],-1,5.5,color='tab:green',linestyle=':')
    ax9.vlines(r3_all[20],-1,5.5,color='tab:green',linestyle=':')
    ax9.vlines(r1_all[18],-1,5.5,color='tab:red',linestyle=':')
    ax9.vlines(r1_all[20],-1,5.5,color='tab:red',linestyle=':')
    ax9.set_xlim(-0.4,1.1)
    ax9.set_xticks([-0.2,0,0.2,0.4,0.6,0.8,1.0])
    ax9.set_xlabel('Correlation coefficient',labelpad=1)
    ax9.set_ylim(-0.3,4.8)
    ax9.set_ylabel('$|\\Delta \\bf{g}|$ (rad)',labelpad=0.1)
    
    # sns.kdeplot(r1, fill=True, color='tab:purple', label='original',ax=ax8) # experiment
    # sns.kdeplot(r3, fill=True, color='tab:orange', label='aligned',ax=ax8)
    sns.ecdfplot(r3, color='k', linestyle='-',linewidth=2.5, label='$SD$ sample',ax=ax8, alpha=0.7, zorder=-1)
    sns.ecdfplot(r1, color='tab:blue', label='original',ax=ax8) # experiment
    sns.ecdfplot(r3, color='tab:orange', linestyle='--', label='aligned',ax=ax8)
    # ax8.annotate('', xy=(-0.3, 1), xytext=(0.6, 1), 
    #             arrowprops=dict(arrowstyle="<->", color='black'),c='black')
    ax8.arrow(r1_all[18],0.3,r3_all[18]-r1_all[18]+0.02,0,facecolor='k',edgecolor='k',linewidth=1.0,length_includes_head=True,head_width=0.0375,head_length=0.075,zorder=5)
    ax8.arrow(r1_all[20],0.99,r3_all[20]-r1_all[20]-0.02,0,facecolor='k',edgecolor='k',linewidth=1.0,length_includes_head=True,head_width=0.0375,head_length=0.075,zorder=5)
    ax8.vlines(r1_all,0,0.1, color='tab:blue',linewidth=0.2,alpha=0.5)
    ax8.vlines(r3_all,0,0.1, color='tab:orange',linewidth=0.2,alpha=0.5)
    ax8.vlines(r1_all[18],0,0.1, color='tab:red',linewidth=1,alpha=1)
    ax8.vlines(r3_all[18],0,0.1, color='tab:green',linewidth=1,alpha=1)
    ax8.vlines(r1_all[20],0,0.1, color='tab:red',linewidth=1,alpha=1)
    ax8.vlines(r3_all[20],0,0.1, color='tab:green',linewidth=1,alpha=1)
    ax8.vlines(r1_all[18],0.1,0.31, linestyle=':',color='tab:red',alpha=1)
    ax8.vlines(r3_all[18],0.1,0.31, linestyle=':',color='tab:green',alpha=1)
    ax8.vlines(r1_all[20],0.1,0.99, linestyle=':',color='tab:red',alpha=1)
    ax8.vlines(r3_all[20],0.1,0.99, linestyle=':',color='tab:green',alpha=1)
    # ax8.legend(facecolor='none',edgecolor='none')
    ax8.set_xlim(-0.4,1.1)
    ax8.set_xticks([-0.2,0,0.2,0.4,0.6,0.8,1.0],[])
    # ax8.set_xlabel('Zero-lag correlation coefficient',labelpad=1)
    ax8.set_ylabel('CDF',labelpad=0.1)
    ax8.set_ylim(0,1.05)
    ax8.set_yticks([0,0.5,1.0])
    ax8.set_title('one session in $GC$-1',pad=3.0)

    axb.scatter(d1, d3, c='tab:orange', alpha=0.5,s=7)
    axb.plot([0,4.5], [0,4.5], c='grey', linewidth=2)
    axb.set_xlim([0, 4.5])
    axb.set_ylim([0, 4.5])
    axb.set_yticks(axb.get_xticks()[:-1])
    axb.set_xlabel('Actual $|\\Delta \\bf{g}|$ (rad)',labelpad=0.1)
    axb.set_ylabel('Estimated $|\\Delta \\bf{g}|$ (rad)',labelpad=0.1)  
    axb.set_title('one session from $GC$-1',pad=2.0)
    axb.text(0.5,4,'$r$ = {:.2f}'.format(pearsonr(d1, d3)[0]))

    with open(path4+'spr_state.pickle', 'rb') as file: #w -> write; b -> binary
         rs=pickle.load(file)
         
    # rs={'Wake':[1,2,3,1],
    #    'REM:'[21,1,2,3],'SWS':[0,24,1,2]}
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
    

    path6='../Data/GC3/predict_d/'  
    np.save(path6+'model_table_gaussian',model_table)
    with open(path6+'spr_state.pickle', 'rb') as file: #w -> write; b -> binary
          rs=pickle.load(file)
    data_d={}
    data_d['OF\n' + r'$\it{GC-1}$']=np.load(path5+'/rsall.npy') 
    data_d['OF\n' + r'$\it{GC-3}$']=rs['OF']
    data_d['WW\n' + r'$\it{GC-3}$']=rs['WW']
    data_d['REM\n' + r'$\it{GC-3}$']=rs['REM']
    data_d['SWS\n' + r'$\it{GC-3}$']=rs['SWS'] 
    cmap = ListedColormap(sns.color_palette('tab20'))
    color_dict = {'OF\n' + r'$\it{GC-1}$':cmap(4),'OF\n' + r'$\it{GC-3}$': cmap(5), 'WW\n' + r'$\it{GC-3}$': cmap(2), 'REM\n' + r'$\it{GC-3}$': cmap(8),'SWS\n' + r'$\it{GC-3}$':cmap(12)} 
    data = []
    for key, values in data_d.items():
        data.extend([(key, value) for value in values])

    print(data_d['OF\n$\\it{GC-1}$'])
    # 使用swarmplot绘制散点图
    color_palette = sns.color_palette('tab10', n_colors=len(data_d))
    sns.swarmplot(x=[item[0] for item in data], y=[item[1] for item in data],palette=color_dict, size=5,ax=axd)
    axd.set_xticklabels(['OF\n' + r'$\it{GC}$-1',
                        'OF\n' + r'$\it{GC}$-3',
                        'WW\n' + r'$\it{GC}$-3',
                        'REM\n' + r'$\it{GC}$-3',
                        'SWS\n' + r'$\it{GC}$-3',
                        ])
    # axd.set_xticklabels(data_d.keys())
    # axd.set_ylabel('Correlation')
    # axd.legend(frameon=False)

    
    # print(list(data_Z3.values()))
    axd.bar(0,np.mean(data_d['OF\n' + r'$\it{GC-1}$']),yerr=np.std(data_d['OF\n' + r'$\it{GC-1}$']),color=cmap(4),alpha=0.5,zorder=5)
    axd.bar(1,np.mean(data_d['OF\n' + r'$\it{GC-3}$']),yerr=np.std(data_d['OF\n' + r'$\it{GC-3}$']),color=cmap(5),alpha=0.5,zorder=5)
    axd.bar(2,np.mean(data_d['WW\n' + r'$\it{GC-3}$']),yerr=np.std(data_d['WW\n' + r'$\it{GC-3}$']),color=cmap(2),alpha=0.5,zorder=5)
    axd.bar(3,np.mean(data_d['REM\n' + r'$\it{GC-3}$']),yerr=np.std(data_d['REM\n' + r'$\it{GC-3}$']),color=cmap(8),alpha=0.5,zorder=5)
    axd.bar(4,np.mean(data_d['SWS\n' + r'$\it{GC-3}$']),yerr=np.std(data_d['SWS\n' + r'$\it{GC-3}$']),color=cmap(12),alpha=0.5,zorder=5)
    

    ax1_pos=ax1.get_position()
    ax4_pos=ax4.get_position()
    ax5_pos=ax5.get_position()
    axa_pos=axa.get_position()
    axc_pos=axc.get_position()

    plt.figtext(ax1_pos.x0-0.02,ax1_pos.y1+0.008,'A',fontsize=15)
    plt.figtext(ax4_pos.x0-0.02,ax4_pos.y1+0.008,'B',fontsize=15)
    plt.figtext(ax5_pos.x0-0.02,ax5_pos.y1+0.008,'C',fontsize=15)
    plt.figtext(axa_pos.x0-0.02,axa_pos.y1+0.008,'D',fontsize=15)
    plt.figtext(axc_pos.x0-0.02,axc_pos.y1+0.008,'E',fontsize=15)


    plt.savefig('../Figures/Figure4.png',format='PNG',dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()