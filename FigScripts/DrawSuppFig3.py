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

def main_HD():
    fig=plt.figure(figsize=(11,8.5))
    IDs = [ 'Mouse12-120807', 'Mouse12-120808']
    # Clus = ['1_8', '1_8', '1_8', '1_8', '1_8', '1_8', '8_11', '5_8']
    States = ['Wake', 'REM', 'SWS']
    StateNames = ['RUN', 'REM', 'SWS']
    sessions=[]
    for i in range(len(IDs)):
        for j in range(len(States)):
            sessions.append(IDs[i]+'_'+States[j])
    no=-1
    pathHD='../Data/HD/ProcessedData/DetectCP_discrete/'
    for no,s in enumerate(sessions):
        ax21=fig.add_axes([0.02,0.85-0.13*no,0.08,0.10])
        ax220=fig.add_axes([0.02+0.12,0.86-0.13*no,0.06,0.08])
        ax22=fig.add_axes([0.02+0.20,0.86-0.13*no,0.06,0.08])
        ax23=fig.add_axes([0.02+0.29,0.86-0.13*no,0.10,0.08])
        ax24=fig.add_axes([0.02+0.42,0.86-0.13*no,0.06,0.08])

        # ax21=fig.add_axes([0.02,0.90-0.16*no,0.15,0.12])
        # ax220=fig.add_axes([0.02+0.23,0.91-0.16*no,0.12,0.10])
        # ax22=fig.add_axes([0.02+0.40,0.91-0.16*no,0.12,0.10])
        # ax23=fig.add_axes([0.02+0.58,0.91-0.16*no,0.20,0.10])
        # ax24=fig.add_axes([0.02+0.84,0.91-0.16*no,0.13,0.10])
        
        with open(pathHD+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        G = output['G']
        c = output['c']
        xlist_w_dict = output['xlist_w_dict']
        node_edgecolors = output['node_edgecolors']
        df = output['df']
        ratios = output['ratios']
        x = output['x']
        # colors = output['colors']
        fpr = output['fpr']
        tpr = output['tpr']
        auc = output['auc']
        
        cmap_Blues = plt.cm.get_cmap('Blues')
        colors_Blues = cmap_Blues(np.linspace(0, 1, 256))
        new_cmap_Blues = mpcolors.ListedColormap(colors_Blues[:144])

        pos = nx.spring_layout(G, k=0.3, iterations=50, weight='weight')  # k参数可以微调节点间的距离

        # print(xlist_w_dict)
        node_edgecolors_name=[]
        zorders=[]
        for i in range(len(node_edgecolors)):
            if node_edgecolors[i]==2:
                node_edgecolors_name.append('orange')
                zorders.append(3)
            else:
                node_edgecolors_name.append('gray')
                zorders.append(1)
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=list(G)[::-1], 
            node_size=30, 
            node_color=list(xlist_w_dict.values())[::-1], 
            cmap=new_cmap_Blues,
            vmin=0,vmax=1.0,
            edgecolors=node_edgecolors_name[::-1],
            linewidths=1.5,
            ax=ax21
        )

        if s[-4:]=='Wake':
            ax21.set_title(r'$\it{HD}$ '+s[:-4]+'RUN', fontsize=6,pad=1)
        else:
            ax21.set_title(r'$\it{HD}$ '+s, fontsize=6,pad=1)

        threshold = 0.4  # 设置阈值
        # print(G.edges(data=True))
        edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        edges_to_draw_weight=[d['weight'] for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw,edge_color=edges_to_draw_weight,edge_cmap=matplotlib.cm.Greys,edge_vmin=0,edge_vmax=1, alpha=1, ax=ax21)
        
        # ax, pos = DrawNetwork(G, c, xlist_w_dict, node_edgecolors, ax21, draw_nodes_kwd={"node_size": 60, "linewidths": 1.5})   
        
        if no==0:
            ax21_position = ax21.get_position()
            cax_width = 0.02  # 定义 colorbar 的宽度
            cax_height = ax21_position.height*0.10  # 使用与 ax21 相同的高度
            cax_x = ax21_position.x1-0.01   # 在 ax21 的右侧偏移一定距离
            cax_y = ax21_position.y0+0.01  # 与 ax21 的底部对齐
            
            cax = fig.add_axes([cax_x, cax_y, cax_width, cax_height])  # Colorbar axes
            # 在 ax21 中绘制数据
            # 添加 colorbar
            cbar = plt.colorbar(cm.ScalarMappable(cmap=new_cmap_Blues),orientation='horizontal', cax=cax, ticks=[0, 1])
            cbar.ax.set_xticklabels(['0', '1'])  # 设置刻度标签
            cbar.set_label('Core score',labelpad=0.1)  # Add label to the colorbar
        
        # ax21.set_title(titles[no], fontsize=8,pad=1)
        ax21.axis('off')  # 关闭坐标轴

        line=sns.kdeplot(data=df, x='Coreness Score', common_norm=False, fill=True, cut=0,ax=ax220)
        # print(line)
        label_patch = mpatches.Patch(color='tab:blue', label='All cells', alpha=1)
        ax220.legend(loc='upper center',handles=[label_patch],facecolor='none',edgecolor='none')
        ax220.set_xlim([0,1])
        ax220.set_xlabel('Core score',labelpad=0.1)
        ax220.set_ylabel('PDF',labelpad=0.1)
        
        # print(df.head(10))
        sns.kdeplot(data=df, x='Coreness Score' , hue='Labels', common_norm=False, fill=True, cut=0, palette=['darkorange','gray'],ax=ax22)

        ax22.axvline(0.6, color='steelblue', linestyle='--', linewidth=1.5)
        

        label1_patch = mpatches.Patch(color='darkorange', label='HDC')
        label2_patch = mpatches.Patch(color='gray', label='nHDC')
        
        ax22.legend(loc='upper center',bbox_to_anchor=(0.38,1.0),handles=[label1_patch, label2_patch],facecolor='none',edgecolor='none')
        ax22.set_xlim([0,1])
        ax22.set_xlabel('Core score',labelpad=0.1)
        ax22.set_ylabel(None)#'PDF',labelpad=0.1)
        
        threshold=0.6
        bars = ax23.bar(x, ratios, color=['darkorange', 'grey', 'darkorange', 'grey'],edgecolor='w',hatch=['','','//','//'],linewidth=1.5,alpha=0.7)
        # ax23.yaxis.grid(True, linestyle='--')
        ax23.spines['right'].set_visible(False)
        ax23.spines['top'].set_visible(False)
        ax23.set_xticks([0.5,2.5])
        ax23.set_xticklabels(['Core','Periphery'])
        ax23.set_ylabel('% of cells in class',labelpad=0.1)
        yticks = [0, 0.25, 0.5, 0.75, 1]
        labels = ['0', '25', '50', '75', '100']
        ax23.set_yticks(yticks)
        ax23.set_yticklabels(labels)
        ax23.set_title('threshold: {:.1f}'.format(threshold))
        # for bar in bars:
        BarID_HD=['HDC','nHDC','HDC','nHDC']
        BarID_GC=['GC','nGC','GC','nGC']
        BarID=[BarID_HD,BarID_GC,BarID_GC]
        for nn,bar in enumerate(bars):
            # print(bar)
            height = bar.get_height()
            
            ax23.annotate(BarID_HD[nn]+'\n{:.1f}%'.format((height*100)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax24.plot(fpr, tpr, c='tab:blue', label='ROC curve (AUC = %0.3f)' % auc)
        ax24.fill_between(fpr, 0, tpr, color='tab:blue',alpha=0.2)
        ax24.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing', c='grey',)
        ax24.set_xlabel('False positive rate',labelpad=0.1)
        ax24.set_ylabel('True positive rate',labelpad=0.1)
        ax24.set_title('AUC: {:.3f}'.format(auc),pad=0.1)
        ax24.set_xlim(-0.03,1)
        ax24.set_ylim(0,1.03)
        
        # if no==0:
        #     ax21_pos = ax21.get_position()
        #     plt.figtext(ax21_pos.x0-0.01,ax21_pos.y1+0.00,'a',fontsize=15)
        #     ax_pos220 = ax220.get_position()
        #     plt.figtext(ax_pos220.x0-0.02,ax21_pos.y1,'b',fontsize=15)
        #     ax_pos22 = ax22.get_position()
        #     plt.figtext(ax_pos22.x0-0.02,ax21_pos.y1,'c',fontsize=15)
        #     ax_pos23 = ax23.get_position()
        #     plt.figtext(ax_pos23.x0-0.02,ax21_pos.y1,'d',fontsize=15)
        #     ax_pos24 = ax24.get_position()
        #     plt.figtext(ax_pos24.x0-0.02,ax21_pos.y1,'e',fontsize=15) 
    
    IDs = [ 'Mouse12-120806', 'Mouse12-120809']
    # Clus = ['1_8', '1_8', '1_8', '1_8', '1_8', '1_8', '8_11', '5_8']
    States = ['Wake', 'REM', 'SWS']
    StateNames = ['RUN', 'REM', 'SWS']
    sessions=[]
    for i in range(len(IDs)):
        for j in range(len(States)):
            sessions.append(IDs[i]+'_'+States[j])
    no=-1
    pathHD='../Data/HD/ProcessedData/DetectCP_discrete/'
    for no,s in enumerate(sessions):
        ax21=fig.add_axes([0.51,0.85-0.13*no,0.08,0.10])
        ax220=fig.add_axes([0.50+0.12,0.86-0.13*no,0.06,0.08])
        ax22=fig.add_axes([0.50+0.20,0.86-0.13*no,0.06,0.08])
        ax23=fig.add_axes([0.50+0.29,0.86-0.13*no,0.10,0.08])
        ax24=fig.add_axes([0.50+0.42,0.86-0.13*no,0.06,0.08])

        # ax21=fig.add_axes([0.02,0.90-0.16*no,0.15,0.12])
        # ax220=fig.add_axes([0.02+0.23,0.91-0.16*no,0.12,0.10])
        # ax22=fig.add_axes([0.02+0.40,0.91-0.16*no,0.12,0.10])
        # ax23=fig.add_axes([0.02+0.58,0.91-0.16*no,0.20,0.10])
        # ax24=fig.add_axes([0.02+0.84,0.91-0.16*no,0.13,0.10])
        
        with open(pathHD+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        G = output['G']
        c = output['c']
        xlist_w_dict = output['xlist_w_dict']
        node_edgecolors = output['node_edgecolors']
        df = output['df']
        ratios = output['ratios']
        x = output['x']
        # colors = output['colors']
        fpr = output['fpr']
        tpr = output['tpr']
        auc = output['auc']
        
        cmap_Blues = plt.cm.get_cmap('Blues')
        colors_Blues = cmap_Blues(np.linspace(0, 1, 256))
        new_cmap_Blues = mpcolors.ListedColormap(colors_Blues[:144])

        pos = nx.spring_layout(G, k=0.3, iterations=50, weight='weight')  # k参数可以微调节点间的距离

        # print(xlist_w_dict)
        node_edgecolors_name=[]
        zorders=[]
        for i in range(len(node_edgecolors)):
            if node_edgecolors[i]==2:
                node_edgecolors_name.append('orange')
                zorders.append(3)
            else:
                node_edgecolors_name.append('gray')
                zorders.append(1)
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=list(G)[::-1], 
            node_size=30, 
            node_color=list(xlist_w_dict.values())[::-1], 
            cmap=new_cmap_Blues,
            vmin=0,vmax=1.0,
            edgecolors=node_edgecolors_name[::-1],
            linewidths=1.5,
            ax=ax21
        )

        if s[-4:]=='Wake':
            ax21.set_title(r'$\it{HD}$ '+s[:-4]+'RUN', fontsize=6,pad=1)
        else:
            ax21.set_title(r'$\it{HD}$ '+s, fontsize=6,pad=1)

        threshold = 0.4  # 设置阈值
        # print(G.edges(data=True))
        edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        edges_to_draw_weight=[d['weight'] for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw,edge_color=edges_to_draw_weight,edge_cmap=matplotlib.cm.Greys,edge_vmin=0,edge_vmax=1, alpha=1, ax=ax21)
        
        # ax, pos = DrawNetwork(G, c, xlist_w_dict, node_edgecolors, ax21, draw_nodes_kwd={"node_size": 60, "linewidths": 1.5})   
        
        if no==0:
            ax21_position = ax21.get_position()
            cax_width = 0.02  # 定义 colorbar 的宽度
            cax_height = ax21_position.height*0.10  # 使用与 ax21 相同的高度
            cax_x = ax21_position.x1-0.01   # 在 ax21 的右侧偏移一定距离
            cax_y = ax21_position.y0+0.01  # 与 ax21 的底部对齐
            
            cax = fig.add_axes([cax_x, cax_y, cax_width, cax_height])  # Colorbar axes
            # 在 ax21 中绘制数据
            # 添加 colorbar
            cbar = plt.colorbar(cm.ScalarMappable(cmap=new_cmap_Blues),orientation='horizontal', cax=cax, ticks=[0, 1])
            cbar.ax.set_xticklabels(['0', '1'])  # 设置刻度标签
            cbar.set_label('Core score',labelpad=0.1)  # Add label to the colorbar
        
        # ax21.set_title(titles[no], fontsize=8,pad=1)
        ax21.axis('off')  # 关闭坐标轴

        line=sns.kdeplot(data=df, x='Coreness Score', common_norm=False, fill=True, cut=0,ax=ax220)
        # print(line)
        label_patch = mpatches.Patch(color='tab:blue', label='All cells', alpha=1)
        ax220.legend(loc='upper center',handles=[label_patch],facecolor='none',edgecolor='none')
        ax220.set_xlim([0,1])
        ax220.set_xlabel('Core score',labelpad=0.1)
        ax220.set_ylabel('PDF',labelpad=0.1)
        
        # print(df.head(10))
        sns.kdeplot(data=df, x='Coreness Score' , hue='Labels', common_norm=False, fill=True, cut=0, palette=['darkorange','gray'],ax=ax22)

        ax22.axvline(0.6, color='steelblue', linestyle='--', linewidth=1.5)
        

        label1_patch = mpatches.Patch(color='darkorange', label='HDC')
        label2_patch = mpatches.Patch(color='gray', label='nHDC')
        
        ax22.legend(loc='upper center',bbox_to_anchor=(0.38,1.0),handles=[label1_patch, label2_patch],facecolor='none',edgecolor='none')
        ax22.set_xlim([0,1])
        ax22.set_xlabel('Core score',labelpad=0.1)
        ax22.set_ylabel(None)#'PDF',labelpad=0.1)
        
        threshold=0.6
        bars = ax23.bar(x, ratios, color=['darkorange', 'grey', 'darkorange', 'grey'],edgecolor='w',hatch=['','','//','//'],linewidth=1.5,alpha=0.7)
        # ax23.yaxis.grid(True, linestyle='--')
        ax23.spines['right'].set_visible(False)
        ax23.spines['top'].set_visible(False)
        ax23.set_xticks([0.5,2.5])
        ax23.set_xticklabels(['Core','Periphery'])
        ax23.set_ylabel('% of cells in class',labelpad=0.1)
        yticks = [0, 0.25, 0.5, 0.75, 1]
        labels = ['0', '25', '50', '75', '100']
        ax23.set_yticks(yticks)
        ax23.set_yticklabels(labels)
        ax23.set_title('threshold: {:.1f}'.format(threshold))
        # for bar in bars:
        BarID_HD=['HDC','nHDC','HDC','nHDC']
        BarID_GC=['GC','nGC','GC','nGC']
        BarID=[BarID_HD,BarID_GC,BarID_GC]
        for nn,bar in enumerate(bars):
            # print(bar)
            height = bar.get_height()
            
            ax23.annotate(BarID_HD[nn]+'\n{:.1f}%'.format((height*100)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax24.plot(fpr, tpr, c='tab:blue', label='ROC curve (AUC = %0.3f)' % auc)
        ax24.fill_between(fpr, 0, tpr, color='tab:blue',alpha=0.2)
        ax24.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing', c='grey',)
        ax24.set_xlabel('False positive rate',labelpad=0.1)
        ax24.set_ylabel('True positive rate',labelpad=0.1)
        ax24.set_title('AUC: {:.3f}'.format(auc),pad=0.1)
        ax24.set_xlim(-0.03,1)
        ax24.set_ylim(0,1.03)
        
        # if no==0:
        #     ax21_pos = ax21.get_position()
        #     plt.figtext(ax21_pos.x0-0.01,ax21_pos.y1+0.00,'a',fontsize=15)
        #     ax_pos220 = ax220.get_position()
        #     plt.figtext(ax_pos220.x0-0.02,ax21_pos.y1,'b',fontsize=15)
        #     ax_pos22 = ax22.get_position()
        #     plt.figtext(ax_pos22.x0-0.02,ax21_pos.y1,'c',fontsize=15)
        #     ax_pos23 = ax23.get_position()
        #     plt.figtext(ax_pos23.x0-0.02,ax21_pos.y1,'d',fontsize=15)
        #     ax_pos24 = ax24.get_position()
        #     plt.figtext(ax_pos24.x0-0.02,ax21_pos.y1,'e',fontsize=15) 


    plt.savefig('../Figures/SuppFigure3_HD_part1.png',format='PNG',dpi=300)
    plt.show()
    plt.close()

    fig=plt.figure(figsize=(11,8.5))
    IDs = [ 'Mouse12-120810', 'Mouse20-130517']
    # Clus = ['1_8', '1_8', '1_8', '1_8', '1_8', '1_8', '8_11', '5_8']
    States = ['Wake', 'REM', 'SWS']
    StateNames = ['RUN', 'REM', 'SWS']
    sessions=[]
    for i in range(len(IDs)):
        for j in range(len(States)):
            sessions.append(IDs[i]+'_'+States[j])
    no=-1
    pathHD='../Data/HD/ProcessedData/DetectCP_discrete/'
    for no,s in enumerate(sessions):
        ax21=fig.add_axes([0.02,0.85-0.13*no,0.08,0.10])
        ax220=fig.add_axes([0.02+0.12,0.86-0.13*no,0.06,0.08])
        ax22=fig.add_axes([0.02+0.20,0.86-0.13*no,0.06,0.08])
        ax23=fig.add_axes([0.02+0.29,0.86-0.13*no,0.10,0.08])
        ax24=fig.add_axes([0.02+0.42,0.86-0.13*no,0.06,0.08])

        # ax21=fig.add_axes([0.02,0.90-0.16*no,0.15,0.12])
        # ax220=fig.add_axes([0.02+0.23,0.91-0.16*no,0.12,0.10])
        # ax22=fig.add_axes([0.02+0.40,0.91-0.16*no,0.12,0.10])
        # ax23=fig.add_axes([0.02+0.58,0.91-0.16*no,0.20,0.10])
        # ax24=fig.add_axes([0.02+0.84,0.91-0.16*no,0.13,0.10])
        
        with open(pathHD+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        G = output['G']
        c = output['c']
        xlist_w_dict = output['xlist_w_dict']
        node_edgecolors = output['node_edgecolors']
        df = output['df']
        ratios = output['ratios']
        x = output['x']
        # colors = output['colors']
        fpr = output['fpr']
        tpr = output['tpr']
        auc = output['auc']
        
        cmap_Blues = plt.cm.get_cmap('Blues')
        colors_Blues = cmap_Blues(np.linspace(0, 1, 256))
        new_cmap_Blues = mpcolors.ListedColormap(colors_Blues[:144])

        pos = nx.spring_layout(G, k=0.3, iterations=50, weight='weight')  # k参数可以微调节点间的距离

        # print(xlist_w_dict)
        node_edgecolors_name=[]
        zorders=[]
        for i in range(len(node_edgecolors)):
            if node_edgecolors[i]==2:
                node_edgecolors_name.append('orange')
                zorders.append(3)
            else:
                node_edgecolors_name.append('gray')
                zorders.append(1)
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=list(G)[::-1], 
            node_size=30, 
            node_color=list(xlist_w_dict.values())[::-1], 
            cmap=new_cmap_Blues,
            vmin=0,vmax=1.0,
            edgecolors=node_edgecolors_name[::-1],
            linewidths=1.5,
            ax=ax21
        )

        if s[-4:]=='Wake':
            ax21.set_title(r'$\it{HD}$ '+s[:-4]+'RUN', fontsize=6,pad=1)
        else:
            ax21.set_title(r'$\it{HD}$ '+s, fontsize=6,pad=1)

        threshold = 0.4  # 设置阈值
        # print(G.edges(data=True))
        edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        edges_to_draw_weight=[d['weight'] for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw,edge_color=edges_to_draw_weight,edge_cmap=matplotlib.cm.Greys,edge_vmin=0,edge_vmax=1, alpha=1, ax=ax21)
        
        # ax, pos = DrawNetwork(G, c, xlist_w_dict, node_edgecolors, ax21, draw_nodes_kwd={"node_size": 60, "linewidths": 1.5})   
        
        if no==0:
            ax21_position = ax21.get_position()
            cax_width = 0.02  # 定义 colorbar 的宽度
            cax_height = ax21_position.height*0.10  # 使用与 ax21 相同的高度
            cax_x = ax21_position.x1-0.01   # 在 ax21 的右侧偏移一定距离
            cax_y = ax21_position.y0+0.01  # 与 ax21 的底部对齐
            
            cax = fig.add_axes([cax_x, cax_y, cax_width, cax_height])  # Colorbar axes
            # 在 ax21 中绘制数据
            # 添加 colorbar
            cbar = plt.colorbar(cm.ScalarMappable(cmap=new_cmap_Blues),orientation='horizontal', cax=cax, ticks=[0, 1])
            cbar.ax.set_xticklabels(['0', '1'])  # 设置刻度标签
            cbar.set_label('Core score',labelpad=0.1)  # Add label to the colorbar
        
        # ax21.set_title(titles[no], fontsize=8,pad=1)
        ax21.axis('off')  # 关闭坐标轴

        line=sns.kdeplot(data=df, x='Coreness Score', common_norm=False, fill=True, cut=0,ax=ax220)
        # print(line)
        label_patch = mpatches.Patch(color='tab:blue', label='All cells', alpha=1)
        ax220.legend(loc='upper center',handles=[label_patch],facecolor='none',edgecolor='none')
        ax220.set_xlim([0,1])
        ax220.set_xlabel('Core score',labelpad=0.1)
        ax220.set_ylabel('PDF',labelpad=0.1)
        
        # print(df.head(10))
        sns.kdeplot(data=df, x='Coreness Score' , hue='Labels', common_norm=False, fill=True, cut=0, palette=['darkorange','gray'],ax=ax22)

        ax22.axvline(0.6, color='steelblue', linestyle='--', linewidth=1.5)
        

        label1_patch = mpatches.Patch(color='darkorange', label='HDC')
        label2_patch = mpatches.Patch(color='gray', label='nHDC')
        
        ax22.legend(loc='upper center',bbox_to_anchor=(0.38,1.0),handles=[label1_patch, label2_patch],facecolor='none',edgecolor='none')
        ax22.set_xlim([0,1])
        ax22.set_xlabel('Core score',labelpad=0.1)
        ax22.set_ylabel(None)#'PDF',labelpad=0.1)
        
        threshold=0.6
        bars = ax23.bar(x, ratios, color=['darkorange', 'grey', 'darkorange', 'grey'],edgecolor='w',hatch=['','','//','//'],linewidth=1.5,alpha=0.7)
        # ax23.yaxis.grid(True, linestyle='--')
        ax23.spines['right'].set_visible(False)
        ax23.spines['top'].set_visible(False)
        ax23.set_xticks([0.5,2.5])
        ax23.set_xticklabels(['Core','Periphery'])
        ax23.set_ylabel('% of cells in class',labelpad=0.1)
        yticks = [0, 0.25, 0.5, 0.75, 1]
        labels = ['0', '25', '50', '75', '100']
        ax23.set_yticks(yticks)
        ax23.set_yticklabels(labels)
        ax23.set_title('threshold: {:.1f}'.format(threshold))
        # for bar in bars:
        BarID_HD=['HDC','nHDC','HDC','nHDC']
        BarID_GC=['GC','nGC','GC','nGC']
        BarID=[BarID_HD,BarID_GC,BarID_GC]
        for nn,bar in enumerate(bars):
            # print(bar)
            height = bar.get_height()
            
            ax23.annotate(BarID_HD[nn]+'\n{:.1f}%'.format((height*100)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax24.plot(fpr, tpr, c='tab:blue', label='ROC curve (AUC = %0.3f)' % auc)
        ax24.fill_between(fpr, 0, tpr, color='tab:blue',alpha=0.2)
        ax24.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing', c='grey',)
        ax24.set_xlabel('False positive rate',labelpad=0.1)
        ax24.set_ylabel('True positive rate',labelpad=0.1)
        ax24.set_title('AUC: {:.3f}'.format(auc),pad=0.1)
        ax24.set_xlim(-0.03,1)
        ax24.set_ylim(0,1.03)
        
        # if no==0:
        #     ax21_pos = ax21.get_position()
        #     plt.figtext(ax21_pos.x0-0.01,ax21_pos.y1+0.00,'a',fontsize=15)
        #     ax_pos220 = ax220.get_position()
        #     plt.figtext(ax_pos220.x0-0.02,ax21_pos.y1,'b',fontsize=15)
        #     ax_pos22 = ax22.get_position()
        #     plt.figtext(ax_pos22.x0-0.02,ax21_pos.y1,'c',fontsize=15)
        #     ax_pos23 = ax23.get_position()
        #     plt.figtext(ax_pos23.x0-0.02,ax21_pos.y1,'d',fontsize=15)
        #     ax_pos24 = ax24.get_position()
        #     plt.figtext(ax_pos24.x0-0.02,ax21_pos.y1,'e',fontsize=15) 
    
    IDs = [ 'Mouse25-140130', 'Mouse28-140313']
    # Clus = ['1_8', '1_8', '1_8', '1_8', '1_8', '1_8', '8_11', '5_8']
    States = ['Wake', 'REM', 'SWS']
    StateNames = ['RUN', 'REM', 'SWS']
    sessions=[]
    for i in range(len(IDs)):
        for j in range(len(States)):
            sessions.append(IDs[i]+'_'+States[j])
    no=-1
    pathHD='../Data/HD/ProcessedData/DetectCP_discrete/'
    for no,s in enumerate(sessions):
        ax21=fig.add_axes([0.51,0.85-0.13*no,0.08,0.10])
        ax220=fig.add_axes([0.50+0.12,0.86-0.13*no,0.06,0.08])
        ax22=fig.add_axes([0.50+0.20,0.86-0.13*no,0.06,0.08])
        ax23=fig.add_axes([0.50+0.29,0.86-0.13*no,0.10,0.08])
        ax24=fig.add_axes([0.50+0.42,0.86-0.13*no,0.06,0.08])

        # ax21=fig.add_axes([0.02,0.90-0.16*no,0.15,0.12])
        # ax220=fig.add_axes([0.02+0.23,0.91-0.16*no,0.12,0.10])
        # ax22=fig.add_axes([0.02+0.40,0.91-0.16*no,0.12,0.10])
        # ax23=fig.add_axes([0.02+0.58,0.91-0.16*no,0.20,0.10])
        # ax24=fig.add_axes([0.02+0.84,0.91-0.16*no,0.13,0.10])
        
        with open(pathHD+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        G = output['G']
        c = output['c']
        xlist_w_dict = output['xlist_w_dict']
        node_edgecolors = output['node_edgecolors']
        df = output['df']
        ratios = output['ratios']
        x = output['x']
        # colors = output['colors']
        fpr = output['fpr']
        tpr = output['tpr']
        auc = output['auc']
        
        cmap_Blues = plt.cm.get_cmap('Blues')
        colors_Blues = cmap_Blues(np.linspace(0, 1, 256))
        new_cmap_Blues = mpcolors.ListedColormap(colors_Blues[:144])

        pos = nx.spring_layout(G, k=0.3, iterations=50, weight='weight')  # k参数可以微调节点间的距离

        # print(xlist_w_dict)
        node_edgecolors_name=[]
        zorders=[]
        for i in range(len(node_edgecolors)):
            if node_edgecolors[i]==2:
                node_edgecolors_name.append('orange')
                zorders.append(3)
            else:
                node_edgecolors_name.append('gray')
                zorders.append(1)
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=list(G)[::-1], 
            node_size=30, 
            node_color=list(xlist_w_dict.values())[::-1], 
            cmap=new_cmap_Blues,
            vmin=0,vmax=1.0,
            edgecolors=node_edgecolors_name[::-1],
            linewidths=1.5,
            ax=ax21
        )

        if s[-4:]=='Wake':
            ax21.set_title(r'$\it{HD}$ '+s[:-4]+'RUN', fontsize=6,pad=1)
        else:
            ax21.set_title(r'$\it{HD}$ '+s, fontsize=6,pad=1)

        threshold = 0.4  # 设置阈值
        # print(G.edges(data=True))
        edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        edges_to_draw_weight=[d['weight'] for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw,edge_color=edges_to_draw_weight,edge_cmap=matplotlib.cm.Greys,edge_vmin=0,edge_vmax=1, alpha=1, ax=ax21)
        
        # ax, pos = DrawNetwork(G, c, xlist_w_dict, node_edgecolors, ax21, draw_nodes_kwd={"node_size": 60, "linewidths": 1.5})   
        
        if no==0:
            ax21_position = ax21.get_position()
            cax_width = 0.02  # 定义 colorbar 的宽度
            cax_height = ax21_position.height*0.10  # 使用与 ax21 相同的高度
            cax_x = ax21_position.x1-0.01   # 在 ax21 的右侧偏移一定距离
            cax_y = ax21_position.y0+0.01  # 与 ax21 的底部对齐
            
            cax = fig.add_axes([cax_x, cax_y, cax_width, cax_height])  # Colorbar axes
            # 在 ax21 中绘制数据
            # 添加 colorbar
            cbar = plt.colorbar(cm.ScalarMappable(cmap=new_cmap_Blues),orientation='horizontal', cax=cax, ticks=[0, 1])
            cbar.ax.set_xticklabels(['0', '1'])  # 设置刻度标签
            cbar.set_label('Core score',labelpad=0.1)  # Add label to the colorbar
        
        # ax21.set_title(titles[no], fontsize=8,pad=1)
        ax21.axis('off')  # 关闭坐标轴

        line=sns.kdeplot(data=df, x='Coreness Score', common_norm=False, fill=True, cut=0,ax=ax220)
        # print(line)
        label_patch = mpatches.Patch(color='tab:blue', label='All cells', alpha=1)
        ax220.legend(loc='upper center',handles=[label_patch],facecolor='none',edgecolor='none')
        ax220.set_xlim([0,1])
        ax220.set_xlabel('Core score',labelpad=0.1)
        ax220.set_ylabel('PDF',labelpad=0.1)
        
        # print(df.head(10))
        sns.kdeplot(data=df, x='Coreness Score' , hue='Labels', common_norm=False, fill=True, cut=0, palette=['darkorange','gray'],ax=ax22)

        ax22.axvline(0.6, color='steelblue', linestyle='--', linewidth=1.5)
        

        label1_patch = mpatches.Patch(color='darkorange', label='HDC')
        label2_patch = mpatches.Patch(color='gray', label='nHDC')
        
        ax22.legend(loc='upper center',bbox_to_anchor=(0.38,1.0),handles=[label1_patch, label2_patch],facecolor='none',edgecolor='none')
        ax22.set_xlim([0,1])
        ax22.set_xlabel('Core score',labelpad=0.1)
        ax22.set_ylabel(None)#'PDF',labelpad=0.1)
        
        threshold=0.6
        bars = ax23.bar(x, ratios, color=['darkorange', 'grey', 'darkorange', 'grey'],edgecolor='w',hatch=['','','//','//'],linewidth=1.5,alpha=0.7)
        # ax23.yaxis.grid(True, linestyle='--')
        ax23.spines['right'].set_visible(False)
        ax23.spines['top'].set_visible(False)
        ax23.set_xticks([0.5,2.5])
        ax23.set_xticklabels(['Core','Periphery'])
        ax23.set_ylabel('% of cells in class',labelpad=0.1)
        yticks = [0, 0.25, 0.5, 0.75, 1]
        labels = ['0', '25', '50', '75', '100']
        ax23.set_yticks(yticks)
        ax23.set_yticklabels(labels)
        ax23.set_title('threshold: {:.1f}'.format(threshold))
        # for bar in bars:
        BarID_HD=['HDC','nHDC','HDC','nHDC']
        BarID_GC=['GC','nGC','GC','nGC']
        BarID=[BarID_HD,BarID_GC,BarID_GC]
        for nn,bar in enumerate(bars):
            # print(bar)
            height = bar.get_height()
            
            ax23.annotate(BarID_HD[nn]+'\n{:.1f}%'.format((height*100)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax24.plot(fpr, tpr, c='tab:blue', label='ROC curve (AUC = %0.3f)' % auc)
        ax24.fill_between(fpr, 0, tpr, color='tab:blue',alpha=0.2)
        ax24.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing', c='grey',)
        ax24.set_xlabel('False positive rate',labelpad=0.1)
        ax24.set_ylabel('True positive rate',labelpad=0.1)
        ax24.set_title('AUC: {:.3f}'.format(auc),pad=0.1)
        ax24.set_xlim(-0.03,1)
        ax24.set_ylim(0,1.03)
        
        # if no==0:
        #     ax21_pos = ax21.get_position()
        #     plt.figtext(ax21_pos.x0-0.01,ax21_pos.y1+0.00,'a',fontsize=15)
        #     ax_pos220 = ax220.get_position()
        #     plt.figtext(ax_pos220.x0-0.02,ax21_pos.y1,'b',fontsize=15)
        #     ax_pos22 = ax22.get_position()
        #     plt.figtext(ax_pos22.x0-0.02,ax21_pos.y1,'c',fontsize=15)
        #     ax_pos23 = ax23.get_position()
        #     plt.figtext(ax_pos23.x0-0.02,ax21_pos.y1,'d',fontsize=15)
        #     ax_pos24 = ax24.get_position()
        #     plt.figtext(ax_pos24.x0-0.02,ax21_pos.y1,'e',fontsize=15) 


    plt.savefig('../Figures/SuppFigure3_HD_part2.png',format='PNG',dpi=300)
    plt.show()
    plt.close()


def main_GC():
    fig=plt.figure(figsize=(11,8.5))
    pathGC1='../Data/GC1/DetectCP_discrete/'
    # sessions=['8a50a33f7fd91df4','1f20835f09e28706','0de4b55d27c9f60f','8f7ddffaf4a5f4c5']
    sessions=['7e888f1d8eaab46b','5b92b96313c3fc19','59825ec5641c94b4','c221438d58a0b796','8a50a33f7fd91df4','1f20835f09e28706','0de4b55d27c9f60f','8f7ddffaf4a5f4c5']
    for no,s in enumerate(sessions):
        if no>=4:
            ax21=fig.add_axes([0.51,0.85-0.13*(no-4),0.08,0.10])
            ax220=fig.add_axes([0.50+0.12,0.86-0.13*(no-4),0.06,0.08])
            ax22=fig.add_axes([0.50+0.20,0.86-0.13*(no-4),0.06,0.08])
            ax23=fig.add_axes([0.50+0.29,0.86-0.13*(no-4),0.10,0.08])
            ax24=fig.add_axes([0.50+0.42,0.86-0.13*(no-4),0.06,0.08])
        else:
            ax21=fig.add_axes([0.02,0.85-0.13*no,0.08,0.10])
            ax220=fig.add_axes([0.02+0.12,0.86-0.13*no,0.06,0.08])
            ax22=fig.add_axes([0.02+0.20,0.86-0.13*no,0.06,0.08])
            ax23=fig.add_axes([0.02+0.29,0.86-0.13*no,0.10,0.08])
            ax24=fig.add_axes([0.02+0.42,0.86-0.13*no,0.06,0.08])

        # ax21=fig.add_axes([0.02,0.90-0.16*no,0.15,0.12])
        # ax220=fig.add_axes([0.02+0.23,0.91-0.16*no,0.12,0.10])
        # ax22=fig.add_axes([0.02+0.40,0.91-0.16*no,0.12,0.10])
        # ax23=fig.add_axes([0.02+0.58,0.91-0.16*no,0.20,0.10])
        # ax24=fig.add_axes([0.02+0.84,0.91-0.16*no,0.13,0.10])
        
        with open(pathGC1+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        G = output['G']
        c = output['c']
        xlist_w_dict = output['xlist_w_dict']
        node_edgecolors = output['node_edgecolors']
        df = output['df']
        ratios = output['ratios']
        x = output['x']
        # colors = output['colors']
        fpr = output['fpr']
        tpr = output['tpr']
        auc = output['auc']
        
        cmap_Blues = plt.cm.get_cmap('Blues')
        colors_Blues = cmap_Blues(np.linspace(0, 1, 256))
        new_cmap_Blues = mpcolors.ListedColormap(colors_Blues[:144])

        pos = nx.spring_layout(G, k=0.3, iterations=50, weight='weight')  # k参数可以微调节点间的距离

        # print(xlist_w_dict)
        node_edgecolors_name=[]
        zorders=[]
        for i in range(len(node_edgecolors)):
            if node_edgecolors[i]==2:
                node_edgecolors_name.append('orange')
                zorders.append(3)
            else:
                node_edgecolors_name.append('gray')
                zorders.append(1)
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=list(G)[::-1], 
            node_size=30, 
            node_color=list(xlist_w_dict.values())[::-1], 
            cmap=new_cmap_Blues,
            vmin=0,vmax=1.0,
            edgecolors=node_edgecolors_name[::-1],
            linewidths=1.5,
            ax=ax21
        )


        threshold = 0.4  # 设置阈值
        # print(G.edges(data=True))
        edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        edges_to_draw_weight=[d['weight'] for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw,edge_color=edges_to_draw_weight,edge_cmap=matplotlib.cm.Greys,edge_vmin=0,edge_vmax=1, alpha=1, ax=ax21)
        
        # ax, pos = DrawNetwork(G, c, xlist_w_dict, node_edgecolors, ax21, draw_nodes_kwd={"node_size": 60, "linewidths": 1.5})   
        ax21.set_title(r'$\it{GC}$-1 OF'+'\n'+s, fontsize=6,pad=1)

        if no==0:
            ax21_position = ax21.get_position()
            cax_width = 0.02  # 定义 colorbar 的宽度
            cax_height = ax21_position.height*0.10  # 使用与 ax21 相同的高度
            cax_x = ax21_position.x1-0.01   # 在 ax21 的右侧偏移一定距离
            cax_y = ax21_position.y0+0.01  # 与 ax21 的底部对齐
            
            cax = fig.add_axes([cax_x, cax_y, cax_width, cax_height])  # Colorbar axes
            # 在 ax21 中绘制数据
            # 添加 colorbar
            cbar = plt.colorbar(cm.ScalarMappable(cmap=new_cmap_Blues),orientation='horizontal', cax=cax, ticks=[0, 1])
            cbar.ax.set_xticklabels(['0', '1'])  # 设置刻度标签
            cbar.set_label('Core score',labelpad=0.1)  # Add label to the colorbar
        
        # ax21.set_title(titles[no], fontsize=8,pad=1)
        ax21.axis('off')  # 关闭坐标轴

        line=sns.kdeplot(data=df, x='Coreness Score', common_norm=False, fill=True, cut=0,ax=ax220)
        # print(line)
        label_patch = mpatches.Patch(color='tab:blue', label='All cells', alpha=1)
        ax220.legend(loc='upper center',handles=[label_patch],facecolor='none',edgecolor='none')
        ax220.set_xlim([0,1])
        ax220.set_xlabel('Core score',labelpad=0.1)
        ax220.set_ylabel('PDF',labelpad=0.1)
        
        # print(df.head(10))
        sns.kdeplot(data=df, x='Coreness Score' , hue='Labels', common_norm=False, fill=True, cut=0, palette=['darkorange','gray'],ax=ax22)

        ax22.axvline(0.6, color='steelblue', linestyle='--', linewidth=1.5)
        

        label1_patch = mpatches.Patch(color='darkorange', label='GC')
        label2_patch = mpatches.Patch(color='gray', label='nGC')
        
        ax22.legend(loc='upper center',bbox_to_anchor=(0.38,1.0),handles=[label1_patch, label2_patch],facecolor='none',edgecolor='none')
        ax22.set_xlim([0,1])
        ax22.set_xlabel('Core score',labelpad=0.1)
        ax22.set_ylabel(None)#'PDF',labelpad=0.1)
        
        threshold=0.6
        bars = ax23.bar(x, ratios, color=['darkorange', 'grey', 'darkorange', 'grey'],edgecolor='w',hatch=['','','//','//'],linewidth=1.5,alpha=0.7)
        # ax23.yaxis.grid(True, linestyle='--')
        ax23.spines['right'].set_visible(False)
        ax23.spines['top'].set_visible(False)
        ax23.set_xticks([0.5,2.5])
        ax23.set_xticklabels(['Core','Periphery'])
        ax23.set_ylabel('% of cells in class',labelpad=0.1)
        yticks = [0, 0.25, 0.5, 0.75, 1]
        labels = ['0', '25', '50', '75', '100']
        ax23.set_yticks(yticks)
        ax23.set_yticklabels(labels)
        ax23.set_title('threshold: {:.1f}'.format(threshold))
        # for bar in bars:
        BarID_HD=['HDC','nHDC','HDC','nHDC']
        BarID_GC=['GC','nGC','GC','nGC']
        BarID=[BarID_HD,BarID_GC,BarID_GC]
        for nn,bar in enumerate(bars):
            # print(bar)
            height = bar.get_height()
            
            ax23.annotate(BarID_GC[nn]+'\n{:.1f}%'.format((height*100)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax24.plot(fpr, tpr, c='tab:blue', label='ROC curve (AUC = %0.3f)' % auc)
        ax24.fill_between(fpr, 0, tpr, color='tab:blue',alpha=0.2)
        ax24.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing', c='grey',)
        ax24.set_xlabel('False positive rate',labelpad=0.1)
        ax24.set_ylabel('True positive rate',labelpad=0.1)
        ax24.set_title('AUC: {:.3f}'.format(auc),pad=0.1)
        ax24.set_xlim(-0.03,1)
        ax24.set_ylim(0,1.03)
        
        # if no==0:
        #     ax21_pos = ax21.get_position()
        #     plt.figtext(ax21_pos.x0-0.01,ax21_pos.y1+0.00,'a',fontsize=15)
        #     ax_pos220 = ax220.get_position()
        #     plt.figtext(ax_pos220.x0-0.02,ax21_pos.y1,'b',fontsize=15)
        #     ax_pos22 = ax22.get_position()
        #     plt.figtext(ax_pos22.x0-0.02,ax21_pos.y1,'c',fontsize=15)
        #     ax_pos23 = ax23.get_position()
        #     plt.figtext(ax_pos23.x0-0.02,ax21_pos.y1,'d',fontsize=15)
        #     ax_pos24 = ax24.get_position()
        #     plt.figtext(ax_pos24.x0-0.02,ax21_pos.y1,'e',fontsize=15) 
    
    pathGC2='../Data/GC2/DetectCP_discrete/'
    sessions = ['Mumbai_1201_1','Kerala_1207_1','Goa_1210_1','Punjab_1217_1']
    for no,s in enumerate(sessions):
        if no>=2:
            ax21=fig.add_axes([0.51,0.30-0.13*(no-2),0.08,0.10])
            ax220=fig.add_axes([0.50+0.12,0.31-0.13*(no-2),0.06,0.08])
            ax22=fig.add_axes([0.50+0.20,0.31-0.13*(no-2),0.06,0.08])
            ax23=fig.add_axes([0.50+0.29,0.31-0.13*(no-2),0.10,0.08])
            ax24=fig.add_axes([0.50+0.42,0.31-0.13*(no-2),0.06,0.08])
        else:
            ax21=fig.add_axes([0.02,0.30-0.13*no,0.08,0.10])
            ax220=fig.add_axes([0.02+0.12,0.31-0.13*no,0.06,0.08])
            ax22=fig.add_axes([0.02+0.20,0.31-0.13*no,0.06,0.08])
            ax23=fig.add_axes([0.02+0.29,0.31-0.13*no,0.10,0.08])
            ax24=fig.add_axes([0.02+0.42,0.31-0.13*no,0.06,0.08])

        # ax21=fig.add_axes([0.02,0.90-0.16*no,0.15,0.12])
        # ax220=fig.add_axes([0.02+0.23,0.91-0.16*no,0.12,0.10])
        # ax22=fig.add_axes([0.02+0.40,0.91-0.16*no,0.12,0.10])
        # ax23=fig.add_axes([0.02+0.58,0.91-0.16*no,0.20,0.10])
        # ax24=fig.add_axes([0.02+0.84,0.91-0.16*no,0.13,0.10])
        
        with open(pathGC2+s+'.pickle', 'rb') as file: #w -> write; b -> binary
            output=pickle.load(file)
        G = output['G']
        c = output['c']
        xlist_w_dict = output['xlist_w_dict']
        node_edgecolors = output['node_edgecolors']
        df = output['df']
        ratios = output['ratios']
        x = output['x']
        # colors = output['colors']
        fpr = output['fpr']
        tpr = output['tpr']
        auc = output['auc']
        
        cmap_Blues = plt.cm.get_cmap('Blues')
        colors_Blues = cmap_Blues(np.linspace(0, 1, 256))
        new_cmap_Blues = mpcolors.ListedColormap(colors_Blues[:144])

        pos = nx.spring_layout(G, k=0.3, iterations=50, weight='weight')  # k参数可以微调节点间的距离

        # print(xlist_w_dict)
        node_edgecolors_name=[]
        zorders=[]
        for i in range(len(node_edgecolors)):
            if node_edgecolors[i]==2:
                node_edgecolors_name.append('orange')
                zorders.append(3)
            else:
                node_edgecolors_name.append('gray')
                zorders.append(1)
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=list(G)[::-1], 
            node_size=30, 
            node_color=list(xlist_w_dict.values())[::-1], 
            cmap=new_cmap_Blues,
            vmin=0,vmax=1.0,
            edgecolors=node_edgecolors_name[::-1],
            linewidths=1.5,
            ax=ax21
        )


        threshold = 0.4  # 设置阈值
        # print(G.edges(data=True))
        edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        edges_to_draw_weight=[d['weight'] for u, v, d in G.edges(data=True) if d['weight'] > threshold]
        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw,edge_color=edges_to_draw_weight,edge_cmap=matplotlib.cm.Greys,edge_vmin=0,edge_vmax=1, alpha=1, ax=ax21)
        
        # ax, pos = DrawNetwork(G, c, xlist_w_dict, node_edgecolors, ax21, draw_nodes_kwd={"node_size": 60, "linewidths": 1.5})   
        ax21.set_title(r'$\it{GC}$-2 1D Track'+'\n'+s, fontsize=6,pad=1)

        if no==0:
            ax21_position = ax21.get_position()
            cax_width = 0.02  # 定义 colorbar 的宽度
            cax_height = ax21_position.height*0.10  # 使用与 ax21 相同的高度
            cax_x = ax21_position.x1-0.01   # 在 ax21 的右侧偏移一定距离
            cax_y = ax21_position.y0+0.01  # 与 ax21 的底部对齐
            
            cax = fig.add_axes([cax_x, cax_y, cax_width, cax_height])  # Colorbar axes
            # 在 ax21 中绘制数据
            # 添加 colorbar
            cbar = plt.colorbar(cm.ScalarMappable(cmap=new_cmap_Blues),orientation='horizontal', cax=cax, ticks=[0, 1])
            cbar.ax.set_xticklabels(['0', '1'])  # 设置刻度标签
            cbar.set_label('Core score',labelpad=0.1)  # Add label to the colorbar
        
        # ax21.set_title(titles[no], fontsize=8,pad=1)
        ax21.axis('off')  # 关闭坐标轴

        line=sns.kdeplot(data=df, x='Coreness Score', common_norm=False, fill=True, cut=0,ax=ax220)
        # print(line)
        label_patch = mpatches.Patch(color='tab:blue', label='All cells', alpha=1)
        ax220.legend(loc='upper center',handles=[label_patch],facecolor='none',edgecolor='none')
        ax220.set_xlim([0,1])
        ax220.set_xlabel('Core score',labelpad=0.1)
        ax220.set_ylabel('PDF',labelpad=0.1)
        
        # print(df.head(10))
        sns.kdeplot(data=df, x='Coreness Score' , hue='Labels', common_norm=False, fill=True, cut=0, palette=['darkorange','gray'],ax=ax22)

        ax22.axvline(0.6, color='steelblue', linestyle='--', linewidth=1.5)
        

        label1_patch = mpatches.Patch(color='darkorange', label='GC')
        label2_patch = mpatches.Patch(color='gray', label='nGC')
        
        ax22.legend(loc='upper center',bbox_to_anchor=(0.38,1.0),handles=[label1_patch, label2_patch],facecolor='none',edgecolor='none')
        ax22.set_xlim([0,1])
        ax22.set_xlabel('Core score',labelpad=0.1)
        ax22.set_ylabel(None)#'PDF',labelpad=0.1)
        
        threshold=0.6
        bars = ax23.bar(x, ratios, color=['darkorange', 'grey', 'darkorange', 'grey'],edgecolor='w',hatch=['','','//','//'],linewidth=1.5,alpha=0.7)
        # ax23.yaxis.grid(True, linestyle='--')
        ax23.spines['right'].set_visible(False)
        ax23.spines['top'].set_visible(False)
        ax23.set_xticks([0.5,2.5])
        ax23.set_xticklabels(['Core','Periphery'])
        ax23.set_ylabel('% of cells in class',labelpad=0.1)
        yticks = [0, 0.25, 0.5, 0.75, 1]
        labels = ['0', '25', '50', '75', '100']
        ax23.set_yticks(yticks)
        ax23.set_yticklabels(labels)
        ax23.set_title('threshold: {:.1f}'.format(threshold))
        # for bar in bars:
        BarID_HD=['HDC','nHDC','HDC','nHDC']
        BarID_GC=['GC','nGC','GC','nGC']
        BarID=[BarID_HD,BarID_GC,BarID_GC]
        for nn,bar in enumerate(bars):
            # print(bar)
            height = bar.get_height()
            
            ax23.annotate(BarID_GC[nn]+'\n{:.1f}%'.format((height*100)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax24.plot(fpr, tpr, c='tab:blue', label='ROC curve (AUC = %0.3f)' % auc)
        ax24.fill_between(fpr, 0, tpr, color='tab:blue',alpha=0.2)
        ax24.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing', c='grey',)
        ax24.set_xlabel('False positive rate',labelpad=0.1)
        ax24.set_ylabel('True positive rate',labelpad=0.1)
        ax24.set_title('AUC: {:.3f}'.format(auc),pad=0.1)
        ax24.set_xlim(-0.03,1)
        ax24.set_ylim(0,1.03)
        
        # if no==0:
        #     ax21_pos = ax21.get_position()
        #     plt.figtext(ax21_pos.x0-0.01,ax21_pos.y1+0.00,'a',fontsize=15)
        #     ax_pos220 = ax220.get_position()
        #     plt.figtext(ax_pos220.x0-0.02,ax21_pos.y1,'b',fontsize=15)
        #     ax_pos22 = ax22.get_position()
        #     plt.figtext(ax_pos22.x0-0.02,ax21_pos.y1,'c',fontsize=15)
        #     ax_pos23 = ax23.get_position()
        #     plt.figtext(ax_pos23.x0-0.02,ax21_pos.y1,'d',fontsize=15)
        #     ax_pos24 = ax24.get_position()
        #     plt.figtext(ax_pos24.x0-0.02,ax21_pos.y1,'e',fontsize=15) 
    
    
    plt.savefig('../Figures/SuppFigure3_GC1&2.png',format='PNG',dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # main_HD()
    main_GC()