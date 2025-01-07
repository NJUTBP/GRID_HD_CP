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
    ax1=fig.add_axes([0.05,0.85,0.20,0.10])
    ax2=fig.add_axes([0.28,0.84,0.17,0.12])

    ax4=fig.add_axes([0.05,0.63,0.18,0.16])
    ax5a=fig.add_axes([0.29,0.63,0.08,0.16])
    ax5b=fig.add_axes([0.38,0.63,0.08,0.16])
    ax6=fig.add_axes([0.05,0.43,0.18,0.16])

    ax7a=fig.add_axes([0.29,0.43,0.08,0.16])
    ax7b=fig.add_axes([0.38,0.43,0.08,0.16])

    # ax8=fig.add_axes([0.51,0.48,0.18,0.16])

    axX_y=0.43
    axX_w=0.085
    axX_w2=0.16
    axX3=fig.add_axes([0.51,axX_y+axX_w+0.01,0.14,axX_w])
    axX6=fig.add_axes([0.51+0.165*1,axX_y+axX_w+0.01,0.14,axX_w])
    axX9=fig.add_axes([0.51+0.165*2,axX_y+axX_w+0.01,0.14,axX_w])
    axX10=fig.add_axes([0.51,axX_y,0.14,axX_w])
    axX11=fig.add_axes([0.51+0.165*1,axX_y,0.14,axX_w])
    axX12=fig.add_axes([0.51+0.165*2,axX_y,0.14,axX_w])
    
    labels = []
    auc = []
    au1 = np.load("../Data/HDAUC.npz", allow_pickle=True)
    au1 = au1['aus']
    for i in range(len(au1)):
        labels.append('HD')
        auc.append(au1[i])
    au1 = np.load("../Data/GC1AUC.npz", allow_pickle=True)
    au1 = au1['aus']
    for i in range(len(au1)):
        labels.append('GC1')
        auc.append(au1[i])
    au1 = np.load("../Data/GC2AUC.npz", allow_pickle=True)
    au1 = au1['aus']
    sele_idx = [1, 2, 4, 6]
    for i in range(len(au1)):
        if i in sele_idx:
            labels.append('GC2')
            auc.append(au1[i])

    data = {'xlist': auc,
            'labels': labels}
    df = pd.DataFrame(data)
    
    # s2=['7e888f1d8eaab46b','0de4b55d27c9f60f']
    s='5b92b96313c3fc19'
    path2='../Data/GC1/core_spatial_info/'

    with open(path2+s+'output.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    grid_score_all=output['grid_score_all']
    mi_rate_all=output['mi_rate_all']
    G=output['G']
    c=output['c']
    xlist_w_dict=output['xlist_w_dict']
    print(xlist_w_dict)
    xstd=output['xstd']
    colornodes=output['colornodes']
    rm=output['rm']
    # core_scores=output['core_scores']
    grid_scores=output['grid_scores']
    grid_socre95=output['grid_socre95']
    # mi_rates=output['mi_rates']
    mi_rates95=output['mi_rates95']
    spatial_coding=output['spatial_coding']

    path3='../Data/GC-1-supp/GS_CS_MI_'
    df_Temp=pd.read_csv(path3+s+'.csv',index_col=0)
    # print(df_Temp)
    core_scores=[]
    mi_rates=[]
    for i in range(16):
        core_scores.append(float(df_Temp[np.abs(df_Temp['GS']-grid_scores[i])<1e-6]['CS']))
        mi_rates.append(float(df_Temp[np.abs(df_Temp['GS']-grid_scores[i])<1e-6]['New_MIR']))
    # print(grid_scores,core_scores)

    # print(xstd)
    # print(colornodes)

    cmap = ListedColormap(sns.color_palette('tab10'))
    color_dict = {'HD': cmap(2), 'GC1': cmap(4), 'GC2': cmap(6)} # purple, pink, blue
    sns.swarmplot(x='labels', y='xlist', data=df, palette=color_dict,ax=ax1)
    ax1.set_xticks(ticks=[0, 1, 2], labels=[r'$\it{HD}$',r'$\it{GC-1}$',r'$\it{GC-2}$'])
    ax1.set_xlabel('')
    ax1.set_ylabel('AUC',labelpad=0.1)
    
    cmap_Blues = plt.cm.get_cmap('Blues')
    colors_Blues = cmap_Blues(np.linspace(0, 1, 256))
    new_cmap_Blues = mpcolors.ListedColormap(colors_Blues[:144])

    pos = nx.spring_layout(G, k=0.3, iterations=50, weight='weight')  # k参数可以微调节点间的距离
    node_edgecolors_name=[]
    for i in range(len(xstd)):
        if xstd[i]==1:
            node_edgecolors_name.append('purple')
        elif xstd[i]==2:
            # print(f'{xlist_w_dict[i]:.2f}')
            node_edgecolors_name.append('orange')
        elif xstd[i]==3:
            node_edgecolors_name.append('tab:pink')
        elif xstd[i]==4:
            node_edgecolors_name.append('green')
        else:
            node_edgecolors_name.append('gray')
    # print(colornodes[:100])
    # print(np.array(colornodes[:100]))
    # print(list(G)[::-1])
    # print(np.array(list(G))[colornodes[:100]])
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=np.array(list(G))[colornodes[:100]], 
        node_size=30, 
        node_color=np.array(list(xlist_w_dict.values()))[colornodes[:100]], 
        cmap=new_cmap_Blues,
        vmin=0,vmax=1.0,
        edgecolors=np.array(node_edgecolors_name)[colornodes[:100]],
        linewidths=1.5,
        ax=ax2
    )
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=np.array(list(G))[colornodes[100:]], 
        node_size=30, 
        node_color=np.array(list(xlist_w_dict.values()))[colornodes[100:]], 
        cmap=new_cmap_Blues,
        vmin=0,vmax=1.0,
        edgecolors=np.array(node_edgecolors_name)[colornodes[100:]],
        linewidths=1.5,
        ax=ax2
    )
    
    threshold = 0.4  # 设置阈值
    # print(G.edges(data=True))
    edges_to_draw = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] > threshold]
    edges_to_draw_weight=[d['weight'] for u, v, d in G.edges(data=True) if d['weight'] > threshold]
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw,edge_color=edges_to_draw_weight,edge_cmap=matplotlib.cm.Greys,edge_vmin=0,edge_vmax=1, alpha=1, ax=ax2)
    ax2.axis('off')
    ax2_pos = ax2.get_position()
    cax_width = 0.008  # 定义 colorbar 的宽度
    cax_height = ax2_pos.height*0.35  # 使用与 ax21 相同的高度
    cax_x = ax2_pos.x1-0.01   # 在 ax21 的右侧偏移一定距离
    cax_y = ax2_pos.y0+0.01  # 与 ax21 的底部对齐
    cax = fig.add_axes([cax_x, cax_y, cax_width, cax_height])  # Colorbar axes
    # 添加 colorbar
    cbar = plt.colorbar(cm.ScalarMappable(cmap=Blues), cax=cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0', '1'])  # 设置刻度标签
    cbar.set_label('Core score', labelpad=2)  # Add label to the colorbar

    #16个神经元放电的exsmaple图
    x_start=0.50
    y_start=0.88
    xp=[]
    # texts=['  low\nnon-GC','  high\nnon-GC','high\n GC','low\nGC']
    # cs=['green', 'orange', 'purple', 'tab:pink']
    for i in range(0, 16):
        # row = (i) // 4  # Calculate the row index
        # col = (i) % 4   # Calculate the column index
        row = (i) % 4  # Calculate the row index
        col = (i) // 4   # Calculate the column index

        # Calculate the coordinates and size of the current subplot
        x = x_start + col * 0.13
        y = y_start - row  * 0.08
        ax3 = fig.add_axes([x, y, 0.08, 0.08])  # Adjust the size as needed
        if i==0:
            ax3_pos = ax3.get_position()
        vmin=0.2
        # vmax=800
        if i<8:
            ax3.imshow(rm[i,:,:],cmap='jet',vmin=0.2*np.min(rm[i,:,:]),vmax=0.8*np.max(rm[i,:,:]))
            # ax3.imshow(rm[i,:,:],cmap='Greens')
        elif i in np.arange(4,8):
            # ax3.imshow(rm[i,:,:],cmap='Oranges',vmax=vmax)
            ax3.imshow(rm[i,:,:],cmap='jet',vmin=0.2*np.min(rm[i,:,:]),vmax=0.8*np.max(rm[i,:,:]))
        elif i in np.arange(8,12):
            # ax3.imshow(rm[i,:,:],cmap='Purples',vmax=vmax)
            ax3.imshow(rm[i,:,:],cmap='jet',vmin=0.2*np.min(rm[i,:,:]),vmax=0.8*np.max(rm[i,:,:]))
        else:
            # ax3.imshow(rm[i,:,:],cmap='PuRd',vmax=vmax)
            # ax3.imshow(rm[i,:,:],cmap='Blues',vmin=0.2*np.min(rm[i,:,:]),vmax=0.8*np.max(rm[i,:,:]))
            if i==15:
                im=ax3.imshow(rm[i,:,:],cmap='jet',vmin=0.2*np.min(rm[i,:,:]),vmax=0.8*np.max(rm[i,:,:]))
                ax_pos = ax3.get_position() 
                # cbar_max=np.nanmax(rm[i,:,:])
                cbar_max=np.max(rm[i,:,:])
                cbar_width = 0.7 * ax_pos.width  # Set the desired width of the colorbar
                cbar_left = ax_pos.x0 + ax_pos.width - cbar_width  # Right align the colorbar
                cax = fig.add_axes([cbar_left, ax_pos.y0-0.009, cbar_width, 0.005])
                cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
                cbar.ax.tick_params(labelsize=5)  
                cbar.set_ticks([0, 0.8*cbar_max])  # 设置刻度位置为最小值和最大值
                cbar.set_ticklabels(['MIN', 'MAX'],fontsize=7)  #
            else:
                ax3.imshow(rm[i,:,:],cmap='jet',vmin=0.2*np.min(rm[i,:,:]),vmax=0.8*np.max(rm[i,:,:]))
                
        #每一列的标题
        # if i%4==0 and i in [0,4]:
        #     plt.figtext(x+0.015,y+0.09,texts[int(i/4)],c=cs[int(i/4)], fontsize=8, fontweight='bold')
        # if i%4==0 and i in [8,12]:
        #     plt.figtext(x+0.03,y+0.09,texts[int(i/4)],c=cs[int(i/4)], fontsize=8, fontweight='bold')    

        text_x = ax3.get_xlim()[1] + 0.06  # X-coordinate for the text#这里似乎有奇怪的bug
        text_y = ax3.get_ylim()[1]     # Starting Y-coordinate for the text
        # print(text_x)
        # Calculate the step size between each number
        step = (ax3.get_ylim()[1] - ax3.get_ylim()[0]) / 3
        step2 = (ax3.get_ylim()[1] - ax3.get_ylim()[0]) / 6
        
        numbers=[core_scores[i],grid_scores[i],mi_rates[i]]
        # Add the numbers and horizontal lines
        NumberName=['$S_\\mathrm{C}$','$S_\\mathrm{G}$','$r_\\mathrm{SI}$']
        for j in range(3):
            number = round(numbers[j], 2)
            if i<4:
                ax3.annotate(NumberName[j]+f'={number:.2f}', (text_x, text_y - (j + 0.5) * step), fontsize=7, va='center')
            else:
                ax3.annotate(f'{number:.2f}', (text_x, text_y - (j + 0.5) * step), fontsize=7, va='center')
            if j!=2:
                ax3.annotate('____', (text_x, text_y - (j + 0.5) * step-step2/2), fontsize=7, va='center')
            # ax3.axhline(y=text_y - (j + 0.5) * step - step2, xmin=text_x - 0.005, xmax=text_x + 0.005, color='black', linewidth=0.5)
        # for j in range(3):
        #     number = round(numbers[j], 2)
        #     ax3.annotate(f'{number:.2f}', (text_x, text_y - j * step), fontsize=5, va='center')
        #     if j != 2:
        #         ax3.annotate('_____', (text_x, text_y- (j + 0.5) * step), fontsize=5, va='center')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_axis_off()
        
        if row==0:
            if col==0:
                ax3.set_title('Periphery\nnGC',color='green',pad=1)
            elif col==1:
                ax3.set_title('Core\nnGC',color='orange',pad=1)
            elif col==2:
                ax3.set_title('Core\nGC',color='purple',pad=1)
            else:
                ax3.set_title('Periphery\nGC',color='tab:pink',pad=1)


    # s='c221438d58a0b796'#'59825ec5641c94b4'#'8a50a33f7fd91df4'#'7e888f1d8eaab46b'#'1f20835f09e28706'#'0de4b55d27c9f60f'#'5b92b96313c3fc19'
    # c22好像不对劲，5982似乎是更好的例子
    path3='../Data/GC-1-supp/GS_CS_MI_'

    df=pd.read_csv(path3+s+'.csv',index_col=0)
    print(df)

    sc=ax4.scatter(df['GS'][::-1],df['CS'][::-1],c=df['New_MIR'][::-1],cmap='jet',s=10)
    ax4.hlines(0.6,-0.6,1.3,color='k',linestyle=':')
    ax4.vlines(grid_socre95,-0.05,1.05,color='k',linestyle=':')
    slope, intercept, r_value, p_value, _ = stats.linregress(df['GS'], df['CS'])
    # 生成拟合的直线数据
    line = slope * np.linspace(-0.55,1.25,91) + intercept
    # 画拟合的直线
    ax4.plot(np.linspace(-0.55,1.25,91), line, 'lightgrey', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
    ax4.set_xlim(-0.6,1.3)
    ax4.set_ylim(-0.05,1.05)
    ax4.set_xlabel('Gridness score',labelpad=0.1)
    ax4.set_ylabel('Core score',labelpad=0.1)
    print('Figure3d',r_value,p_value)
    
    ax4_pos = ax4.get_position()
    cbar_width = 0.7 * ax4_pos.width  # Set the desired width of the colorbar
    cbar_left = ax4_pos.x0 + ax4_pos.width - cbar_width  # Right align the colorbar
    cax = fig.add_axes([cbar_left, ax4_pos.y0+ax4_pos.height+0.003, cbar_width, 0.005])
    cbar = fig.colorbar(sc, cax=cax, orientation='horizontal',ticklocation='top')
    # cax.set_xticks([np.log10(10), np.log10(25), np.log10(50),np.log10(100),np.log10(200)])
    cax.tick_params(pad=0.1)
    # cax.set_xticklabels([10,25,50,100,200])
    cax.set_xlabel('Spatial information rate (bit/s)', labelpad=2)
    
    color_dict = {'NGC': 'tab:olive', 'GC': 'tab:cyan'}
    sns.stripplot(data=df, x="GCG", y="New_MIR", order=['GC', 'NGC'], palette=color_dict, ax=ax5a, alpha=0.5)
    sns.boxplot(data=df, x="GCG", y="New_MIR", order=['GC', 'NGC'], palette=color_dict, ax=ax5a, boxprops={'facecolor':'None'}, showfliers=False)
    pairs = [("GC", "NGC")]
    annotator = Annotator(ax5a, pairs, data=df, x="GCG", y="New_MIR", order=['GC', 'NGC'],)
    annotator.configure(test='t-test_ind', text_format='star', line_height=0.03, line_width=1,fontsize=9)
    annotator.apply_and_annotate()

    color_dict = {'Core': 'tab:red', 'Peri': 'tab:blue'}
    sns.stripplot(data=df, x="CPG", y="New_MIR", order=['Core', 'Peri'], palette=color_dict, ax=ax5b, alpha=0.5)
    sns.boxplot(data=df, x="CPG", y="New_MIR", order=['Core', 'Peri'], palette=color_dict, ax=ax5b, boxprops={'facecolor':'None'}, showfliers=False)
    pairs = [("Core", "Peri")]
    annotator = Annotator(ax5b, pairs, data=df, x="CPG", y="New_MIR", order=['Core', 'Peri'],)
    annotator.configure(test='t-test_ind', text_format='star', line_height=0.03, line_width=1,fontsize=9)
    annotator.apply_and_annotate()

    ax5a.set_xticks([0,1])
    ax5a.set_xticklabels(['GCs','nGCs'])
    ax5a.set_xlabel('')
    ax5a.set_ylim(0,8)
    ax5a.set_yticks([0,2,4,6,8])
    ax5a.set_ylabel('Spatial information rate (bit/s)')
    ax5a.spines["top"].set_visible(False)
    ax5a.spines["right"].set_visible(False)

    ax5b.set_xticks([0,1])
    ax5b.set_xticklabels(['Core','Periphery'])
    ax5b.set_xlabel('')
    ax5b.set_ylim(0,8)
    ax5b.set_yticks([0,2,4,6,8])
    ax5b.set_yticklabels([])
    ax5b.set_ylabel('')
    ax5b.spines["top"].set_visible(False)
    ax5b.spines["right"].set_visible(False)

    print(df.head(10))
    print(df[(df['New_MIR']>4)*(df['GCCPG']=='PeriGC')])
    # newdf=df.copy(deep=True)
    newdf=df.drop(df[(df['New_MIR']>4)*(df['GCCPG']=='PeriGC')].index)
    color_dict = {'PeriNGC': 'green', 'CoreNGC': 'orange', 'CoreGC': 'purple', 'PeriGC': 'tab:pink'}
    sns.stripplot(data=df, x="GCCPG", y="New_MIR", order=['PeriNGC', 'CoreNGC', 'CoreGC', 'PeriGC'], palette=color_dict, ax=ax6, alpha=0.5)
    sns.boxplot(data=df, x="GCCPG", y="New_MIR", order=['PeriNGC', 'CoreNGC', 'CoreGC', 'PeriGC'], palette=color_dict, ax=ax6, boxprops={'facecolor':'None'}, showfliers=False)
    pairs = [('PeriNGC', 'CoreNGC'), ('CoreNGC', 'CoreGC'), ('CoreGC', 'PeriGC'), ('PeriNGC', 'PeriGC'),('CoreNGC', 'PeriGC')]
    # annotator = Annotator(ax6, pairs, data=newdf, x="GCCPG", y="New_MIR", order=['PeriNGC', 'CoreNGC', 'CoreGC', 'PeriGC'])
    # annotator.configure(test='t-test_ind', text_format='star', line_height=0.02, line_width=1)
    # annotator.apply_and_annotate()
    ax6.plot([0,0,1,1],[7.1,7.3,7.3,7.1], color='k', linewidth=1)
    ax6.plot([1,1,2,2],[7.6,7.8,7.8,7.6], color='k', linewidth=1)
    ax6.plot([2,2,3,3],[8.1,8.3,8.3,8.1], color='k', linewidth=1)
    ax6.plot([1,1,3,3],[8.8,9.0,9.0,8.8], color='k', linewidth=1)
    ax6.plot([0,0,3,3],[9.5,9.7,9.7,9.5], color='k', linewidth=1)
    ax6.text(0.5,7.4, '****',ha='center',fontsize=9)
    ax6.text(1.5,7.95, 'ns',ha='center',fontsize=9)
    ax6.text(2.5,8.2, '**',ha='center',fontsize=9)
    ax6.text(2.0,9.05, '**',ha='center',fontsize=9)
    ax6.text(1.5,9.85, 'ns',ha='center',fontsize=9)

    ax6.set_xticks([0,1,2,3],['Periphery\nnGC','Core\nnGC','Core\nGC','Periphery\nGC'])
    ax6.set_xlabel('')
    ax6.set_ylim(0,9.75)
    ax6.set_yticks([0,2,4,6,8])
    ax6.set_ylabel('Spatial information rate (bit/s)',labelpad=2)
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    color_dict = {'NGC': 'tab:olive', 'GC': 'tab:cyan'}
    sns.stripplot(data=df, x="GCG", y="r_1st_2nd_RM", order=['GC', 'NGC'], palette=color_dict, ax=ax7a, alpha=0.5)
    sns.boxplot(data=df, x="GCG", y="r_1st_2nd_RM", order=['GC', 'NGC'], palette=color_dict, ax=ax7a, boxprops={'facecolor':'None'}, showfliers=False)
    pairs = [("GC", "NGC")]
    annotator = Annotator(ax7a, pairs, data=df, x="GCG", y="r_1st_2nd_RM", order=['GC', 'NGC'],)
    annotator.configure(test='t-test_ind', text_format='star', line_height=0.03, line_width=1,fontsize=9)
    annotator.apply_and_annotate()

    color_dict = {'Core': 'tab:red', 'Peri': 'tab:blue'}
    sns.stripplot(data=df, x="CPG", y="r_1st_2nd_RM", order=['Core', 'Peri'], palette=color_dict, ax=ax7b, alpha=0.5)
    sns.boxplot(data=df, x="CPG", y="r_1st_2nd_RM", order=['Core', 'Peri'], palette=color_dict, ax=ax7b, boxprops={'facecolor':'None'}, showfliers=False)
    pairs = [("Core", "Peri")]
    annotator = Annotator(ax7b, pairs, data=df, x="CPG", y="r_1st_2nd_RM", order=['Core', 'Peri'],)
    annotator.configure(test='t-test_ind', text_format='star', line_height=0.03, line_width=1,fontsize=9)
    annotator.apply_and_annotate()

    ax7a.set_xticks([0,1])
    ax7a.set_xticklabels(['GCs','nGCs'])
    ax7a.set_xlabel('')
    ax7a.set_ylabel('Correlation between the\n1st- and 2nd-half rate map',labelpad=0.1)
    ax7a.spines["top"].set_visible(False)
    ax7a.spines["right"].set_visible(False)

    ax7b.set_xticks([0,1])
    ax7b.set_xticklabels(['Core','Periphery'])
    ax7b.set_xlabel('')
    ax7b.set_yticklabels([])
    ax7b.set_ylabel('')
    ax7b.spines["top"].set_visible(False)
    ax7b.spines["right"].set_visible(False)

    # color_dict = {'PeriNGC': 'green', 'CoreNGC': 'orange', 'CoreGC': 'purple', 'PeriGC': 'tab:pink'}
    # sns.stripplot(data=df, x="GCCPG", y="r_1st_2nd_RM", order=['PeriNGC', 'CoreNGC', 'CoreGC', 'PeriGC'], palette=color_dict, ax=ax8, alpha=0.5)
    # sns.boxplot(data=df, x="GCCPG", y="r_1st_2nd_RM", order=['PeriNGC', 'CoreNGC', 'CoreGC', 'PeriGC'], palette=color_dict, ax=ax8, boxprops={'facecolor':'None'}, showfliers=False)
    # pairs = [('PeriNGC', 'CoreNGC'), ('CoreNGC', 'CoreGC'), ('CoreGC', 'PeriGC'), ('PeriNGC', 'PeriGC')]
    # annotator = Annotator(ax8, pairs, data=df, x="GCCPG", y="r_1st_2nd_RM", order=['PeriNGC', 'CoreNGC', 'CoreGC', 'PeriGC'])
    # annotator.configure(test='t-test_ind', text_format='star', line_height=0.03, line_width=1)
    # annotator.apply_and_annotate()

    pathX_HD='../Data/HD/ProcessedData/MI_core/'
    pathX_GC1='../Data/GC1/MI_core/'
    pathX_GC2='../Data/GC2/MI_core/'
    info_all_HD=np.load(pathX_HD+'info_all.npy',allow_pickle=True).item()
    info_all_GC1=np.load(pathX_GC1+'info_all.npy',allow_pickle=True).item()
    info_all_GC2=np.load(pathX_GC2+'info_all.npy',allow_pickle=True).item()
    core_score_HD=info_all_HD['Mouse12-120806_Wake']['core_score']
    mi_rate_HD=info_all_HD['Mouse12-120806_Wake']['mi_rate']
    axX3.scatter(core_score_HD,mi_rate_HD,s=20,facecolor='tab:red',edgecolor='none',alpha=0.5)
    slope, intercept, r_value, p_value, _ = stats.linregress(core_score_HD, mi_rate_HD)
    # 生成拟合的直线数据
    line = slope * np.array(core_score_HD) + intercept
    # 画拟合的直线
    axX3.plot(core_score_HD, line, 'k', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.5)
    # axX3.set_xticklabels([])
    axX3.set_xlabel('Core score')
    # axX3.set_ylabel('Spatial information rate',labelpad=0.1)
    axX3pos=axX3.get_position()
    plt.figtext(axX3pos.x0-0.025,axX3pos.y0-0.005,'Spatial information rate (bit/s)',fontsize=7.5,rotation=90,ha='center',va='center')
    axX3.set_title(r'$\mathit{HD}$',pad=1)
    axX3.set_xlim(-0.05,1.05)
    axX3.set_xticklabels([])
    # axX3_pos = axX3.get_position()  
    # plt.figtext(axX3_pos.x0-0.03,ax51_pos.y1+0.01,'f',fontsize=abc_size)
    print(p_value)
    # axX3.text(0.15, 4, '****', fontsize=12, fontweight='bold')

    core_score_GC1=info_all_GC1[s]['core_score']#'0de4b55d27c9f60f'
    mi_rate_GC1=info_all_GC1[s]['mi_rate']#'0de4b55d27c9f60f'
    # fmt = '{:.0f}'.format
    # tick_formatter = mtick.FuncFormatter(lambda x, pos: fmt(x))
    # 使用这个格式化函数来改变y轴标签的格式
    # axX6.yaxis.set_major_formatter(tick_formatter)
    axX6.scatter(core_score_GC1,mi_rate_GC1,s=20,facecolor='tab:red',edgecolor='none',alpha=0.5)
    slope, intercept, r_value, p_value, _ = stats.linregress(core_score_GC1, mi_rate_GC1)
    # 生成拟合的直线数据
    line = slope * np.array(core_score_GC1) + intercept
    # 画拟合的直线
    axX6.plot(core_score_GC1, line, 'k', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.5)
    axX6.set_xlabel('Core score')
    # axX6.set_xticklabels([])
    axX6.set_title(r'$\mathit{GC}$-1',pad=1)
    axX6.set_xlim(-0.05,1.05)
    axX6.set_xticklabels([])
    print(p_value)
    # axX6.text(0.2, 60, '****', fontsize=12, fontweight='bold')

    fmt = '{:.1f}'.format
    tick_formatter = mtick.FuncFormatter(lambda x, pos: fmt(x))
    core_score_GC2=info_all_GC2['Mumbai_1201_1']['core_score']
    mi_rate_GC2=info_all_GC2['Mumbai_1201_1']['mi_rate']
    axX9.scatter(core_score_GC2,mi_rate_GC2,s=20,facecolor='tab:red',edgecolor='none',alpha=0.5)
    slope, intercept, r_value, p_value, _ = stats.linregress(core_score_GC2, mi_rate_GC2)
    # 生成拟合的直线数据
    line = slope * np.array(core_score_GC2) + intercept
    # 画拟合的直线
    axX9.plot(core_score_GC2, line, 'k', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.5)
    axX9.set_xlabel('Core score')
    # axX9.set_xticklabels([])
    axX9.set_title(r'$\mathit{GC}$-2',pad=1)
    axX9.set_ylim(-0.1,2.2)
    axX9.set_xlim(-0.05,1.05)
    axX9.set_xticklabels([])
    print(p_value)
    # axX9.text(0.2, 1.5, '****', fontsize=12, fontweight='bold')

    #axX10=fig.add_axes([0.51,axX_y-axX_w2-0.02,axX_w4*1.5,axX_w4]) 
    info_all_HD=np.load(pathX_HD+'info_all.npy',allow_pickle=True).item()
    IDs = ['Mouse12-120806', 'Mouse12-120807', 'Mouse12-120808', 'Mouse12-120809', 'Mouse12-120810',
          'Mouse20-130517', 'Mouse28-140313', 'Mouse25-140130']
    Clus = ['1_8', '1_8', '1_8', '1_8', '1_8', '1_8', '8_11', '5_8']
    States = ['Wake', 'REM', 'SWS']
    for m in ['Mouse12-120807','Mouse28-140313','Mouse12-120808' ]:
        for s in States:
            core_score=info_all_HD[m+'_'+s]['core_score']
            mi_rate=info_all_HD[m+'_'+s]['mi_rate']
            slope, intercept, r_value, p_value, _ = stats.linregress(core_score, mi_rate)
            # 生成拟合的直线数据
            line = slope * np.linspace(np.min(core_score),np.max(core_score),300) + intercept
            # 画拟合的直线
            if p_value<0.01:
                if s=='Wake':   
                    axX10.plot(np.linspace(np.min(core_score),np.max(core_score),300), line, linestyle='-',linewidth=1.0,c='tab:blue', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.3)
                elif s=='REM':   
                    axX10.plot(np.linspace(np.min(core_score),np.max(core_score),300), line,  linestyle='-',linewidth=1.0,c='tab:orange', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.3)
                else:
                    axX10.plot(np.linspace(np.min(core_score),np.max(core_score),300), line,  linestyle='-',linewidth=1.0,c='tab:green', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.3)
            else:
                print('ERRRRR'+m+s)
    # Set the legend labels and colors
    labels = ['RUN', 'REM', 'SWS']
    colors = {'RUN': 'brown', 'REM': 'forestgreen', 'SWS': 'steelblue'}

    # Create the legend with custom colors
    legend = axX10.legend(labels, labels=[f'{label}' for label in labels],facecolor='none',edgecolor='none')
    # axX10.set_xlabel('Core score')
    # axX10.set_ylabel('Spatial information rate',labelpad=0.1)

    # Adjust the font size of the legend
    plt.rcParams['legend.fontsize'] = 5

    # Optional: Customize the legend appearance
    # legend.get_frame().set_facecolor('white')  # Set legend background color
    # legend.get_frame().set_edgecolor('grey')    # Set legend border color
    # legend.get_frame().set_linewidth(1)  # Set legend border width

    axX10.set_xlim(-0.05,1.05)

    info_all_GC1=np.load(pathX_GC1+'info_all.npy',allow_pickle=True).item()
    session_list=['8a50a33f7fd91df4', '1f20835f09e28706', '8f7ddffaf4a5f4c5', '7e888f1d8eaab46b', '5b92b96313c3fc19', '59825ec5641c94b4', 'c221438d58a0b796']
    for s in session_list:
        
        print(s)
        core_score=info_all_GC1[s]['core_score']
        mi_rate=info_all_GC1[s]['mi_rate']
        
        core_score_s, mi_rate_s, _, _=ransac_filter(core_score, mi_rate)
        # core_score_s, mi_rate_s=np.array(core_score), np.array(mi_rate)
        slope, intercept, r_value, p_value, _ = stats.linregress(list(core_score_s.flatten()), list(mi_rate_s.flatten()))
        # slope, intercept, r_value, p_value, _ = stats.linregress(core_score, core_score)
        # 生成拟合的直线数据
        line = slope * np.linspace(np.min(core_score_s),np.max(core_score_s),300) + intercept
        # 画拟合的直线
        
        if p_value<0.01:
            axX11.plot(np.linspace(np.min(core_score_s),np.max(core_score_s),300), line, 'tab:red', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.5)
        else:
            axX11.plot(np.linspace(np.min(core_score_s),np.max(core_score_s),300), line, 'grey', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.5)
    # axX11.set_xticklabels([])
    axX11.set_xlabel('Core score',labelpad=0.1)
    axX11.yaxis.set_major_formatter(tick_formatter)
      
    axX11.set_xlim(-0.05,1.05)

    info_all_GC2=np.load(pathX_GC2+'info_all.npy',allow_pickle=True).item()
    session_list= ['Kerala_1207_1','Goa_1210_1','Punjab_1217_1']
    for s in session_list:
        core_score=info_all_GC2[s]['core_score']
        mi_rate=info_all_GC2[s]['mi_rate']
        
        core_score_s, mi_rate_s, _, _=ransac_filter(core_score, mi_rate)
        # core_score_s, mi_rate_s=np.array(core_score), np.array(mi_rate)
        slope, intercept, r_value, p_value, _ = stats.linregress(list(core_score_s.flatten()), list(mi_rate_s.flatten()))
        # slope, intercept, r_value, p_value, _ = stats.linregress(core_score, core_score)
        # 生成拟合的直线数据
        # line = slope * np.array(core_score_s) + intercept
        # # 画拟合的直线
        # if p_value<0.01:
        #     axX12.plot(core_score_s, line, 'tab:red', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.5)
        # else:
        #     axX12.plot(core_score_s, line, 'grey', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.5)
        line = slope * np.linspace(np.min(core_score_s),np.max(core_score_s),300) + intercept
        # 画拟合的直线
        if p_value<0.01:

            axX12.plot(np.linspace(np.min(core_score_s),np.max(core_score_s),300), line, 'tab:red', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.5)
        else:
            axX12.plot(np.linspace(np.min(core_score_s),np.max(core_score_s),300), line, 'grey', label='y={:.2f}x+{:.2f}'.format(slope,intercept),alpha=0.5)
    
    # axX12.set_xticklabels([])
    # axX12.set_xlabel('Core score')
    fmt = '{:.1f}'.format
    tick_formatter = mtick.FuncFormatter(lambda x, pos: fmt(x))
    axX12.yaxis.set_major_formatter(tick_formatter)
    axX12.set_xlim(-0.05,1.05)

    axes = [axX3, axX10, axX6, axX11, axX9, axX12]
    # 遍历列表，设置每个子图的ytick的pad
    for ax in axes:
        ax.tick_params(axis='y', which='major', pad=0.1)  # 将pad设置为较小的值，例如2



    ax1_pos = ax1.get_position()
    plt.figtext(ax1_pos.x0-0.03,ax1_pos.y1+0.015,'A',fontsize=15)
    ax2_pos = ax2.get_position()  
    plt.figtext(ax2_pos.x0-0.02,ax1_pos.y1+0.015,'B',fontsize=15)
    plt.figtext(x_start-0.02,ax1_pos.y1+0.015,'C',fontsize=15)
    ax4_pos = ax4.get_position()  
    plt.figtext(ax1_pos.x0-0.03,ax4_pos.y1+0.01,'D',fontsize=15)
    ax5a_pos = ax5a.get_position()  
    plt.figtext(ax2_pos.x0-0.02,ax5a_pos.y1+0.01,'E',fontsize=15)
    ax6_pos = ax6.get_position()  
    plt.figtext(ax6_pos.x0-0.03,ax6_pos.y1+0.01,'F',fontsize=15)
    ax7a_pos = ax7a.get_position()  
    plt.figtext(ax7a_pos.x0-0.02,ax7a_pos.y1+0.01,'G',fontsize=15)
    axX3_pos = axX3.get_position()  
    plt.figtext(x_start-0.02,axX3_pos.y1+0.01,'H',fontsize=15)


    plt.savefig('../Figures/Figure3.png',format='PNG',dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()