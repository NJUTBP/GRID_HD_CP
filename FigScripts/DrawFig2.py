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
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colors as mpcolors

from matplotlib.cm import Blues
from matplotlib import cm  # Import the cm module

import random
import networkx as nx
import sys
sys.path.append('../') # 将上级目录加入 sys.path
from DrawStandard import *
import pickle
from DetectCP import GenerateGraph, CalCoreness, Metrics, DrawNetwork
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
    with open('../Data/fig2a.pickle', 'rb') as file: #w -> write; b -> binary
        output=pickle.load(file)
    smoothed_data=output['smoothed_datas'] # 8个神经元的放电数据
    pink_line_height=0.7
    # arr=np.array([0,1,0,0,1,1,0,0,0,1,0,1,1,1,0,1])
    
    fig=plt.figure(figsize=(8.5,11))
    ax11=fig.add_axes([0.055,0.88,0.10*11/8.5,0.10])
    ax12=fig.add_axes([0.055,0.77,0.08*11/8.5,0.08])
    ax13=fig.add_axes([0.87,0.89,0.08*11/8.5,0.08])

    ax14_y=0.77
    ax14_x=0.24
    ax14_w=0.10*11/8.5
    ax14=fig.add_axes([ax14_x, ax14_y,0.08*11/8.5,0.08]) # 前两个参数是子图左下角的坐标
    ax15=fig.add_axes([ax14_x+ax14_w, ax14_y,0.08*11/8.5,0.08])
    ax16=fig.add_axes([ax14_x+2*ax14_w, ax14_y,0.08*11/8.5,0.08])
    ax17=fig.add_axes([ax14_x+3*ax14_w+0.05, ax14_y,0.08*11/8.5,0.08])
    axarr=fig.add_axes([ax14_x,ax14_y+0.092,3.8*ax14_w+0.05,0.04-0.017])
    
    ax18a=fig.add_axes([0.30, 0.88, 0.01, 0.10])
    ax18b=fig.add_axes([0.36, 0.93, 0.10*11/8.5, 0.01*8.5/11])
    ax19=fig.add_axes([0.56, 0.89, 0.08*11/8.5,0.08])
    
    arr = np.arange(12)
    np.random.shuffle(arr)
    rearr=np.zeros(12).astype(np.int64)
    for i in range(12):
        rearr[int(arr[i])]=i
    print(arr,rearr)
    for no in range(12):
        # print(no)
        ax11.plot(smoothed_data[arr[no],:] + no*pink_line_height , color='grey')
        ax11.text(-6, 0.1*np.max(smoothed_data[no,:]) + no*pink_line_height, str(12-no), color='k',ha='right',fontsize=6)
        
    ax11.axis('off')
    # ax_pos = ax11.get_position()  
    # plt.figtext(ax_pos.x0-0.01,ax_pos.y1+0.03,'a',fontsize=15)
    # plt.figtext(ax_pos.x0-0.005,ax_pos.y1+0.009, r'$\mathrm{I}$',fontsize=10)

    im2=output['im2']
    for i in range(im2.shape[0]):
        im2[i,i]=np.nan
    cmap_oranges = plt.cm.get_cmap('Oranges')
    cmap_binary = plt.cm.get_cmap('binary')
    colors_oranges = cmap_oranges(np.linspace(0, 1, 256))
    colors_binary = cmap_binary(np.linspace(0, 1, 256))
    combined_colors = np.vstack((colors_binary[0], colors_oranges))
    new_cmap_oranges = mpcolors.ListedColormap(combined_colors)
    
    im2s=im2[arr,:][:,arr]
    im=ax12.imshow(im2s,cmap=new_cmap_oranges)
    ax_pos = ax12.get_position() 
    cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0, 0.005, 0.5*ax_pos.height]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)  
    cbar.set_ticks([0, 1])  # 设置刻度位置为最小值和最大值
    # cbar.set_ticklabels(['MIN', 'MAX'])  #
    ax12.set_xticks(np.arange(12))
    ax12.set_yticks(np.arange(12))
    ax12.set_xticklabels(np.arange(12)+1,fontsize=6)
    ax12.set_yticklabels(np.arange(12)+1,fontsize=6)
    ax12.set_title('$A$',fontsize=9)
    # ax_pos = ax12.get_position()  
    # plt.figtext(ax_pos.x0-0.007,ax_pos.y1+0.007, r'$\mathrm{III}$',fontsize=10)

    # im2s=im2[arr,:][:,arr]
    im=ax13.imshow(im2[:12,:][:,:12],cmap=new_cmap_oranges)
    ax_pos = ax13.get_position() 
    cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0, 0.005, 0.5*ax_pos.height]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)  
    cbar.set_ticks([0, 1])  # 设置刻度位置为最小值和最大值
    # cbar.set_ticklabels(['MIN', 'MAX'])  #
    ax13.set_xticks(np.arange(12))
    ax13.set_yticks(np.arange(12))
    ax13.set_xticklabels(rearr+1,fontsize=6)
    ax13.set_yticklabels(rearr+1,fontsize=6)
    ax13.set_title('Shuffle with maximal $R_{\\alpha,\\beta}$',pad=0.1)
    rect = mpatches.Rectangle((-0.5, -0.5), 7, 7, linewidth=1, edgecolor='k', facecolor='none')
    ax13.add_patch(rect)
    
    # ax_pos = ax13.get_position()
    # plt.figtext(ax_pos.x0-0.007,ax_pos.y1+0.007, r'$\mathrm{III}$',fontsize=10)

    # ax_pos14 = ax14.get_position()  
    # plt.figtext(ax_pos14.x0,ax_pos14.y1,'Shuffle',fontsize=10)
    # ax18_pos = ax18.get_position()
    # start = (ax18_pos.x0 + ax18_pos.width / 2, ax18_pos.y0+0.008)
    # arrow = patches.FancyArrow(start[0], start[1], -0.013, -0.018, color='k', width=0.001)
    # ax_arrow.add_patch(arrow)
    # plt.figtext(ax18_pos.x0 + ax18_pos.width / 2-0.005,ax18_pos.y0-0.007,r'$\odot$',fontsize=10)
    # plt.figtext(ax18_pos.x0-0.03,ax18_pos.y1-0.005, r'$\mathrm{IV}$',fontsize=rm_size)

    arr_shuffle=np.arange(12)
    np.random.shuffle(arr_shuffle)
    im2s=im2[arr[arr_shuffle],:][:,arr[arr_shuffle]]
    ax14.imshow(im2s,cmap=new_cmap_oranges)
    ax14.set_xticks(np.arange(12))
    ax14.set_yticks(np.arange(12))
    ax14.set_xticklabels(arr_shuffle+1,fontsize=6)
    ax14.set_yticklabels(arr_shuffle+1,fontsize=6)
    ax14.set_title('Shuffle #1          $R_1$',pad=0.1)
    np.random.shuffle(arr_shuffle)
    im2s=im2[arr[arr_shuffle],:][:,arr[arr_shuffle]]
    ax15.imshow(im2s,cmap=new_cmap_oranges)
    ax15.set_xticks(np.arange(12))
    ax15.set_yticks(np.arange(12))
    ax15.set_xticklabels(arr_shuffle+1,fontsize=6)
    ax15.set_yticklabels(arr_shuffle+1,fontsize=6)
    ax15.set_title('Shuffle #2          $R_2$',pad=0.1)
    np.random.shuffle(arr_shuffle)
    im2s=im2[arr[arr_shuffle],:][:,arr[arr_shuffle]]
    ax16.imshow(im2s,cmap=new_cmap_oranges)
    ax16.set_xticks(np.arange(12))
    ax16.set_yticks(np.arange(12))
    ax16.set_title('Shuffle #3          $R_3$',pad=0.1)
    ax16.set_xticklabels(arr_shuffle+1,fontsize=6)
    ax16.set_yticklabels(arr_shuffle+1,fontsize=6)
    np.random.shuffle(arr_shuffle)
    im2s=im2[arr[arr_shuffle],:][:,arr[arr_shuffle]]
    im=ax17.imshow(im2s,cmap=new_cmap_oranges)
    ax17.set_xticks(np.arange(12))
    ax17.set_yticks(np.arange(12))
    ax17.set_title('Shuffle #M          $R_M$',pad=0.1)
    ax17.set_xticklabels(arr_shuffle+1,fontsize=6)
    ax17.set_yticklabels(arr_shuffle+1,fontsize=6)
    ax_pos = ax17.get_position() 
    cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0+ 0.003, 0.005, 0.5*ax_pos.height]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)  
    cbar.set_ticks([0, 1])  # 设置刻度位置为最小值和最大值
    # cbar.set_ticklabels(['MIN', 'MAX'])  

    alpha=0.92
    beta=5/12
    Cstar=np.zeros(12)
    Cstar[:5]=np.arange(1,6)*(1-alpha)/2/beta
    Cstar[5:]=(np.arange(6,13)-beta)*(1-alpha)/2/(12-beta)+(1+alpha)/2
    Cstar=Cstar.reshape((1,12))[:,::-1]
    print(Cstar)

    ax18a.imshow(Cstar.T,cmap='Greys')
    ax18b.imshow(Cstar,cmap='Greys')

    ax18a.text(-0.5,7,'$\\downarrow$',color='tab:red',ha='center',va='center',fontsize=15)
    ax18a.text(1.5,7,'$\\alpha$: gradient',color='tab:red',ha='center',va='center',fontsize=6,rotation=90)
    ax18a.text(0.1,2.5,'Core neurons',color='w',ha='center',va='center',fontsize=6,rotation=90)
    ax18a.set_xticks([])
    # ax18a.set_ylim(-0.5,11.5)
    ax18a.set_yticks([0,6,7,11])
    ax18a.set_yticklabels(['$N$ = 12','[$\\beta N$]+1','[$\\beta N$]','1'],color='tab:blue')
    ax18a.set_title('$P(\\alpha,\\beta)$',fontsize=9)

    ax18b.text(-3,0,'$\\times$',fontsize=15,ha='center',va='center')
    ax18b.text(15,0,'$\\rightarrow$',fontsize=15,ha='center',va='center')
    ax18b.set_yticks([])
    ax18b.set_xticks([0,6,7,11])
    ax18b.set_xticklabels([])
    ax18b.set_title('$P^{T}$',fontsize=9)

    matrix=Cstar.T*Cstar
    # print()
    im=ax19.imshow(matrix,cmap='Greys')
    ax19.set_xticks([0,6,7,11])
    ax19.set_yticks([0,6,7,11])
    ax19.set_xticklabels([])
    ax19.set_yticklabels([])
    ax19.set_title('$B$',fontsize=9)
    ax_pos = ax19.get_position() 
    cax = fig.add_axes([ax_pos.x1 + 0.0045, ax_pos.y0, 0.005, 0.5*ax_pos.height]) 
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=5)  
    cbar.set_ticks([0, 1])  # 设置刻度位置为最小值和最大值
    # cbar.set_ticklabels(['MIN', 'MAX'])  #

    # ax14_y=0.77
    # ax14_x=0.24
    # ax14_w=0.10*11/8.5
    # ax14=fig.add_axes([ax14_x, ax14_y,0.08*11/8.5,0.08])
    # axarr=fig.add_axes([ax14_x,ax14_y+0.08,3.8*ax14_w+0.05,0.04])

    # axarr.arrow()
    # axarr.arrow(0.375,1,ax14_w/2-0.375,0-1,head_length=0.06,head_width=0.02,length_includes_head=True,edgecolor='k',facecolor='k')
    # axarr.arrow(0.375,1,ax14_w/2+ax14_w-0.375,0-1,head_length=0.06,head_width=0.02,length_includes_head=True,edgecolor='k',facecolor='k')
    # axarr.arrow(0.375,1,ax14_w/2+2*ax14_w-0.375,0-1,head_length=0.06,head_width=0.02,length_includes_head=True,edgecolor='k',facecolor='k')
    # axarr.arrow(0.375,1,ax14_w/2+3*ax14_w+0.05-0.375,0-1,head_length=0.06,head_width=0.02,length_includes_head=True,edgecolor='k',facecolor='k')
    axarr.annotate('',xy=(ax14_w/2,0),xytext=(0.375,1),arrowprops=dict(width=0.05,headwidth=3,headlength=6,facecolor='tab:orange',edgecolor='tab:orange', shrink=0.05))
    axarr.annotate('',xy=(ax14_w/2+ax14_w,0),xytext=(0.375,1),arrowprops=dict(width=0.05,headwidth=3,headlength=6,facecolor='tab:orange',edgecolor='tab:orange', shrink=0.05))
    axarr.annotate('',xy=(ax14_w/2+2*ax14_w,0),xytext=(0.375,1),arrowprops=dict(width=0.05,headwidth=3,headlength=6,facecolor='tab:orange',edgecolor='tab:orange', shrink=0.05))
    axarr.annotate('',xy=(ax14_w/2+3*ax14_w+0.05,0),xytext=(0.375,1),arrowprops=dict(width=0.05,headwidth=3,headlength=6,facecolor='tab:orange',edgecolor='tab:orange', shrink=0.05))
    # axarr.text(0.35,0.05,'Summation of $\\odot$',color='tab:orange')
    axarr.text(0.39,-0.10,'Summation of \n elementwise production',color='tab:orange',ha='center')
    axarr.axis('off')

    axarr.set_xlim(0,3.8*ax14_w+0.05)
    axarr.set_ylim(-0.1,1.1)

    # labels = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_M$']
    # eq_symbol = r'$=$'  # LaT
    # for ax, label in zip([ax14, ax15, ax16, ax17], labels):
    #     ax.text(5.5,15, '||\n'+label, fontsize=7, ha='center', va='center')
    ax17.text(-5.5,5,'......',fontsize=15, ha='center', va='center')

    ax13.text(5.5,18,'$\\downarrow$',fontsize=15, ha='center', va='center')
    ax13.text(5.5,21,'Iterating over $\\alpha$ and $\\beta$,\nrepeating Steps III-V',fontsize=7.5, ha='center', va='center')
    ax13.text(5.5,25,'$\\downarrow$',fontsize=15, ha='center', va='center')
    ax13.text(5.5,29,'$S_{C,i}=Z\\sum_{\\alpha,\\beta} P_i(\\alpha,\\beta)R_{\\alpha,\\beta}$',fontsize=9, ha='center', va='center')
    
    ax11_pos = ax11.get_position()  
    plt.figtext(0.01,ax11_pos.y1,'A',fontsize=15)
    plt.figtext(ax11_pos.x0-0.005,ax11_pos.y1+0.005, r'$\mathrm{i}$',fontsize=10)
    ax12_pos = ax12.get_position()
    plt.figtext(ax12_pos.x0-0.005,ax12_pos.y1+0.005, r'$\mathrm{ii}$',fontsize=10)
    ax13_pos = ax13.get_position()
    plt.figtext(ax13_pos.x0-0.035,ax13_pos.y1+0.005, r'$\mathrm{v}$',fontsize=10)
    plt.figtext(ax13_pos.x0-0.035,ax12_pos.y1+0.005, r'$\mathrm{vi}$',fontsize=10)
    ax14_pos = ax14.get_position()
    plt.figtext(ax14_pos.x0-0.020,ax14_pos.y1+0.005, r'$\mathrm{iv}$',fontsize=10)
    ax18a_pos = ax18a.get_position()
    plt.figtext(ax18a_pos.x0-0.05,ax18a_pos.y1+0.005, r'$\mathrm{iii}$',fontsize=10)
    
    pathHD='../Data/HD/ProcessedData/DetectCP/'
    pathGC1='../Data/GC1/DetectCP/'
    pathGC2='../Data/GC2/DetectCP/'
    path=[pathHD,pathGC1,pathGC2]
    titles=[r'one session in $\it{HD}$',r'one session in $\it{GC}$-1',r'one session in $\it{GC}$-2']
    
    for no,p in enumerate(path):
        ax21=fig.add_axes([0.02,0.60-0.16*no,0.15,0.12])
        ax220=fig.add_axes([0.02+0.23,0.61-0.16*no,0.12,0.10])
        ax22=fig.add_axes([0.02+0.40,0.61-0.16*no,0.12,0.10])
        ax23=fig.add_axes([0.02+0.58,0.61-0.16*no,0.20,0.10])
        ax24=fig.add_axes([0.02+0.84,0.61-0.16*no,0.13,0.10])
        
        with open(p+'output.pickle', 'rb') as file: #w -> write; b -> binary
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
        
        if no==0:
            ax21_position = ax21.get_position()
            cax_width = 0.008  # 定义 colorbar 的宽度
            cax_height = ax21_position.height*0.40  # 使用与 ax21 相同的高度
            cax_x = ax21_position.x1-0.008   # 在 ax21 的右侧偏移一定距离
            cax_y = ax21_position.y0+0.003  # 与 ax21 的底部对齐
            
            cax = fig.add_axes([cax_x, cax_y, cax_width, cax_height])  # Colorbar axes
            # 在 ax21 中绘制数据
            # 添加 colorbar
            cbar = plt.colorbar(cm.ScalarMappable(cmap=new_cmap_Blues), cax=cax, ticks=[0, 1])
            cbar.ax.set_yticklabels(['0', '1'])  # 设置刻度标签
            cbar.set_label('Core score',labelpad=2)  # Add label to the colorbar
        
        ax21.set_title(titles[no], fontsize=8,pad=1)
        ax21.axis('off')  # 关闭坐标轴

        line=sns.kdeplot(data=df, x='Coreness Score', common_norm=False, fill=True, cut=0,ax=ax220)
        # print(line)
        label_patch = mpatches.Patch(color='tab:blue', label='All cells', alpha=1)
        ax220.legend(loc='upper center',handles=[label_patch],facecolor='none',edgecolor='none',fontsize=7.5)
        ax220.set_xlim([0,1])
        ax220.set_xlabel('Core score',labelpad=1)
        ax220.set_ylabel('PDF',labelpad=1)
        
        
        if no!=1:
            sns.kdeplot(data=df, x='Coreness Score' , hue='Labels', common_norm=False, fill=True, cut=0, palette=['darkorange','gray'],ax=ax22)
            # ax22.hist(df[df['Labels']=='o']['Coreness Score'],bins=np.linspace(0,1,11),color='gray',alpha=0.5)
            # ax22.hist(df[df['Labels']=='h']['Coreness Score'],bins=np.linspace(0,1,11),color='darkorange',alpha=0.5)
        else:
            sns.kdeplot(data=df, x='Coreness Score' , hue='Labels', common_norm=False, fill=True, cut=0, palette=['gray','darkorange'],ax=ax22)
            # print(df.head(10))
            # ax22.hist(df[df['Labels']=='g']['Coreness Score'])
        if no==0:
            ax22.vlines(0.6, 0, 23, color='steelblue', linestyle='--', linewidth=1.5)
        elif no==1:
            ax22.vlines(0.6, 0, 1.4, color='steelblue', linestyle='--', linewidth=1.5)
        else:
            ax22.vlines(0.6, 0, 2.2, color='steelblue', linestyle='--', linewidth=1.5)
        
        if no==0:
            label1_patch = mpatches.Patch(color='darkorange', label='HDC')
            label2_patch = mpatches.Patch(color='gray', label='nHDC')
        else:
            label1_patch = mpatches.Patch(color='darkorange', label='GC')
            label2_patch = mpatches.Patch(color='gray', label='nGC')
        
        ax22.legend(loc='upper center',bbox_to_anchor=(0.38,1.0),handles=[label1_patch, label2_patch],facecolor='none',edgecolor='none',fontsize=7.5)
        ax22.set_xlim([0,1])
        ax22.set_xlabel('Core score',labelpad=1)
        ax22.set_ylabel('PDF',labelpad=1)
        
        threshold=0.6
        bars = ax23.bar(x, ratios, color=['darkorange', 'grey', 'darkorange', 'grey'],edgecolor='w',hatch=['','','//','//'],linewidth=1.5,alpha=0.7)
        # ax23.yaxis.grid(True, linestyle='--')
        ax23.spines['right'].set_visible(False)
        ax23.spines['top'].set_visible(False)
        ax23.set_xticks([0.5,2.5])
        ax23.set_xticklabels(['Core\nneurons','Periphery\nneurons'])
        ax23.set_ylabel('% of cells in class',labelpad=1)
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
            
            ax23.annotate(BarID[no][nn]+'\n{:.1f}%'.format((height*100)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",fontsize=7.5,
                        ha='center', va='bottom')
        
        ax24.plot(fpr, tpr, c='tab:blue', label='ROC curve (AUC = %0.3f)' % auc)
        ax24.fill_between(fpr, 0, tpr, color='tab:blue',alpha=0.2)
        ax24.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing', c='grey',)
        if no==0:
            ax24.set_xlabel('False positive rate',labelpad=1)
            ax24.set_ylabel('True positive rate',labelpad=1)
        else:
            ax24.set_xlabel('False positive rate',labelpad=1)
            ax24.set_ylabel('True positive rate',labelpad=1)
        ax24.set_title('AUC: {:.3f}'.format(auc))
        ax24.set_xlim(-0.03,1)
        ax24.set_ylim(0,1.03)

        if no==0:
            ax21_pos = ax21.get_position()
            plt.figtext(ax21_pos.x0-0.01,ax21_pos.y1+0.00,'B',fontsize=15)
            ax_pos220 = ax220.get_position()
            plt.figtext(ax_pos220.x0-0.02,ax21_pos.y1,'C',fontsize=15)
            ax_pos22 = ax22.get_position()
            plt.figtext(ax_pos22.x0-0.02,ax21_pos.y1,'D',fontsize=15)
            ax_pos23 = ax23.get_position()
            plt.figtext(ax_pos23.x0-0.02,ax21_pos.y1,'E',fontsize=15)
            ax_pos24 = ax24.get_position()
            plt.figtext(ax_pos24.x0-0.02,ax21_pos.y1,'F',fontsize=15) 

    plt.savefig('../Figures/Figure2.png',format='PNG',dpi=300)
    plt.show()
    plt.close()

if __name__ == '__main__':
        main()